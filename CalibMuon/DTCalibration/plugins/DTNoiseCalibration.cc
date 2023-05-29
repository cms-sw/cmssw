/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 *          A. Vilela Pereira - INFN Torino 
 */

#include "DTNoiseCalibration.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// Digis
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

// Database
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include "TH2F.h"
#include "TFile.h"

using namespace edm;
using namespace std;

DTNoiseCalibration::DTNoiseCalibration(const edm::ParameterSet& pset)
    : digiToken_(consumes<DTDigiCollection>(pset.getParameter<InputTag>("digiLabel"))),
      useTimeWindow_(pset.getParameter<bool>("useTimeWindow")),
      triggerWidth_(pset.getParameter<double>("triggerWidth")),
      timeWindowOffset_(pset.getParameter<int>("timeWindowOffset")),
      maximumNoiseRate_(pset.getParameter<double>("maximumNoiseRate")),
      useAbsoluteRate_(pset.getParameter<bool>("useAbsoluteRate")),
      readDB_(true),
      defaultTtrig_(0),
      //fastAnalysis_( pset.getParameter<bool>("fastAnalysis", true) ),
      wireIdWithHisto_(std::vector<DTWireId>()),
      lumiMax_(3000),
      dtGeomToken_(esConsumes<edm::Transition::BeginRun>()),
      ttrigToken_(
          esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", pset.getUntrackedParameter<string>("dbLabel")))) {
  // Get the debug parameter for verbose output
  //debug = ps.getUntrackedParameter<bool>("debug");
  /*// The analysis type
  // The wheel & sector interested for the time-dependent analysis
  wh = ps.getUntrackedParameter<int>("wheel", 0);
  sect = ps.getUntrackedParameter<int>("sector", 6);*/

  if (pset.exists("defaultTtrig")) {
    readDB_ = false;
    defaultTtrig_ = pset.getParameter<int>("defaultTtrig");
  }

  if (pset.exists("cellsWithHisto")) {
    vector<string> cellsWithHisto = pset.getParameter<vector<string> >("cellsWithHisto");
    for (vector<string>::const_iterator cell = cellsWithHisto.begin(); cell != cellsWithHisto.end(); ++cell) {
      //FIXME: Use regex to check whether format is right
      if ((!cell->empty()) && (*cell) != "None") {
        stringstream linestr;
        int wheel, station, sector, sl, layer, wire;
        linestr << (*cell);
        linestr >> wheel >> station >> sector >> sl >> layer >> wire;
        wireIdWithHisto_.push_back(DTWireId(wheel, station, sector, sl, layer, wire));
      }
    }
  }

  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName", "noise.root");
  rootFile_ = new TFile(rootFileName.c_str(), "RECREATE");
  rootFile_->cd();
}

void DTNoiseCalibration::beginJob() {
  LogVerbatim("Calibration") << "[DTNoiseCalibration]: Begin job";

  nevents_ = 0;

  TH1::SetDefaultSumw2(true);
  int numBin = (triggerWidth_ * 32 / 25) / 50;
  hTDCTriggerWidth_ = new TH1F("TDC_Time_Distribution", "TDC_Time_Distribution", numBin, 0, triggerWidth_ * 32 / 25);
}

void DTNoiseCalibration::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  // Get the DT Geometry
  dtGeom_ = setup.getHandle(dtGeomToken_);
  // tTrig
  if (readDB_)
    tTrigMap_ = &setup.getData(ttrigToken_);
  runBeginTime_ = time_t(run.beginTime().value() >> 32);
  runEndTime_ = time_t(run.endTime().value() >> 32);
}

void DTNoiseCalibration::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  ++nevents_;

  // Get the digis from the event
  const Handle<DTDigiCollection>& dtdigis = event.getHandle(digiToken_);

  time_t eventTime = time_t(event.time().value() >> 32);
  unsigned int lumiSection = event.luminosityBlock();

  // Loop over digis
  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It = dtdigis->begin(); dtLayerId_It != dtdigis->end(); ++dtLayerId_It) {
    // Define time window
    float upperLimit = 0.;
    if (useTimeWindow_) {
      if (readDB_) {
        float tTrig, tTrigRMS, kFactor;
        DTSuperLayerId slId = ((*dtLayerId_It).first).superlayerId();
        int status = tTrigMap_->get(slId, tTrig, tTrigRMS, kFactor, DTTimeUnits::counts);
        if (status != 0)
          throw cms::Exception("DTNoiseCalibration") << "Could not find tTrig entry in DB for" << slId << endl;
        upperLimit = tTrig - timeWindowOffset_;
      } else {
        upperLimit = defaultTtrig_ - timeWindowOffset_;
      }
    }

    double triggerWidth_s = 0.;
    if (useTimeWindow_)
      triggerWidth_s = ((upperLimit * 25) / 32) / 1e9;
    else
      triggerWidth_s = double(triggerWidth_ / 1e9);
    LogTrace("Calibration") << ((*dtLayerId_It).first).superlayerId() << " Trigger width (s): " << triggerWidth_s;

    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
         digiIt != ((*dtLayerId_It).second).second;
         ++digiIt) {
      //Check the TDC trigger width
      int tdcTime = (*digiIt).countsTDC();
      if (!useTimeWindow_) {
        if ((((float)tdcTime * 25) / 32) > triggerWidth_) {
          LogError("Calibration") << "Digi has a TDC time (ns) higher than the pre-defined TDC trigger width: "
                                  << ((float)tdcTime * 25) / 32;
          continue;
        }
      }

      hTDCTriggerWidth_->Fill(tdcTime);

      if (useTimeWindow_ && tdcTime > upperLimit)
        continue;

      const DTLayerId dtLId = (*dtLayerId_It).first;
      const DTTopology& dtTopo = dtGeom_->layer(dtLId)->specificTopology();
      const int firstWire = dtTopo.firstChannel();
      const int lastWire = dtTopo.lastChannel();
      const int nWires = lastWire - firstWire + 1;

      // Book the occupancy histos
      if (theHistoOccupancyMap_.find(dtLId) == theHistoOccupancyMap_.end()) {
        string histoName = "DigiOccupancy_" + getLayerName(dtLId);
        rootFile_->cd();
        TH1F* hOccupancyHisto = new TH1F(histoName.c_str(), histoName.c_str(), nWires, firstWire, lastWire + 1);
        LogTrace("Calibration") << "  Created occupancy Histo: " << hOccupancyHisto->GetName();
        theHistoOccupancyMap_[dtLId] = hOccupancyHisto;
      }
      theHistoOccupancyMap_[dtLId]->Fill((*digiIt).wire(), 1. / triggerWidth_s);

      // Histos vs lumi section
      const DTChamberId dtChId = dtLId.chamberId();
      if (chamberOccupancyVsLumiMap_.find(dtChId) == chamberOccupancyVsLumiMap_.end()) {
        string histoName = "OccupancyVsLumi_" + getChamberName(dtChId);
        rootFile_->cd();
        TH1F* hOccupancyVsLumiHisto = new TH1F(histoName.c_str(), histoName.c_str(), lumiMax_, 0, lumiMax_);
        LogTrace("Calibration") << "  Created occupancy histo: " << hOccupancyVsLumiHisto->GetName();
        chamberOccupancyVsLumiMap_[dtChId] = hOccupancyVsLumiHisto;
      }
      chamberOccupancyVsLumiMap_[dtChId]->Fill(lumiSection, 1. / triggerWidth_s);

      const DTWireId wireId(dtLId, (*digiIt).wire());
      if (find(wireIdWithHisto_.begin(), wireIdWithHisto_.end(), wireId) != wireIdWithHisto_.end()) {
        if (theHistoOccupancyVsLumiMap_.find(wireId) == theHistoOccupancyVsLumiMap_.end()) {
          string histoName = "OccupancyVsLumi_" + getChannelName(wireId);
          rootFile_->cd();
          TH1F* hOccupancyVsLumiHisto = new TH1F(histoName.c_str(), histoName.c_str(), lumiMax_, 0, lumiMax_);
          LogTrace("Calibration") << "  Created occupancy histo: " << hOccupancyVsLumiHisto->GetName();
          theHistoOccupancyVsLumiMap_[wireId] = hOccupancyVsLumiHisto;
        }
        theHistoOccupancyVsLumiMap_[wireId]->Fill(lumiSection, 1. / triggerWidth_s);
      }

      // Histos vs time
      if (chamberOccupancyVsTimeMap_.find(dtChId) == chamberOccupancyVsTimeMap_.end()) {
        string histoName = "OccupancyVsTime_" + getChamberName(dtChId);
        float secPerBin = 20.0;
        unsigned int nBins = ((unsigned int)(runEndTime_ - runBeginTime_)) / secPerBin;
        rootFile_->cd();
        TH1F* hOccupancyVsTimeHisto = new TH1F(
            histoName.c_str(), histoName.c_str(), nBins, (unsigned int)runBeginTime_, (unsigned int)runEndTime_);
        for (int k = 0; k < hOccupancyVsTimeHisto->GetNbinsX(); ++k) {
          if (k % 10 == 0) {
            unsigned int binLowEdge = hOccupancyVsTimeHisto->GetBinLowEdge(k + 1);
            time_t timeValue = time_t(binLowEdge);
            hOccupancyVsTimeHisto->GetXaxis()->SetBinLabel((k + 1), ctime(&timeValue));
          }
        }
        size_t lastBin = hOccupancyVsTimeHisto->GetNbinsX();
        unsigned int binUpperEdge =
            hOccupancyVsTimeHisto->GetBinLowEdge(lastBin) + hOccupancyVsTimeHisto->GetBinWidth(lastBin);
        time_t timeValue = time_t(binUpperEdge);
        hOccupancyVsTimeHisto->GetXaxis()->SetBinLabel((lastBin), ctime(&timeValue));

        LogTrace("Calibration") << "  Created occupancy histo: " << hOccupancyVsTimeHisto->GetName();
        chamberOccupancyVsTimeMap_[dtChId] = hOccupancyVsTimeHisto;
      }
      chamberOccupancyVsTimeMap_[dtChId]->Fill((unsigned int)eventTime, 1. / triggerWidth_s);
    }
  }
}

void DTNoiseCalibration::endJob() {
  LogVerbatim("Calibration") << "[DTNoiseCalibration] Total number of events analyzed: " << nevents_;

  // Save the TDC digi plot
  rootFile_->cd();
  hTDCTriggerWidth_->Write();

  double normalization = 1. / double(nevents_);

  for (map<DTWireId, TH1F*>::const_iterator wHisto = theHistoOccupancyVsLumiMap_.begin();
       wHisto != theHistoOccupancyVsLumiMap_.end();
       ++wHisto) {
    (*wHisto).second->Scale(normalization);
    (*wHisto).second->Write();
  }

  for (map<DTChamberId, TH1F*>::const_iterator chHisto = chamberOccupancyVsLumiMap_.begin();
       chHisto != chamberOccupancyVsLumiMap_.end();
       ++chHisto) {
    (*chHisto).second->Scale(normalization);
    (*chHisto).second->Write();
  }

  for (map<DTChamberId, TH1F*>::const_iterator chHisto = chamberOccupancyVsTimeMap_.begin();
       chHisto != chamberOccupancyVsTimeMap_.end();
       ++chHisto) {
    (*chHisto).second->Scale(normalization);
    (*chHisto).second->Write();
  }

  // Save on file the occupancy histos and write the list of noisy cells
  DTStatusFlag statusMap;
  for (map<DTLayerId, TH1F*>::const_iterator lHisto = theHistoOccupancyMap_.begin();
       lHisto != theHistoOccupancyMap_.end();
       ++lHisto) {
    if ((*lHisto).second) {
      (*lHisto).second->Scale(normalization);
      rootFile_->cd();
      (*lHisto).second->Write();
      const DTTopology& dtTopo = dtGeom_->layer((*lHisto).first)->specificTopology();
      const int firstWire = dtTopo.firstChannel();
      const int lastWire = dtTopo.lastChannel();
      const int nWires = lastWire - firstWire + 1;
      // Find average in layer
      double averageRate = 0.;
      for (int bin = 1; bin <= (*lHisto).second->GetNbinsX(); ++bin)
        averageRate += (*lHisto).second->GetBinContent(bin);

      if (nWires)
        averageRate /= nWires;
      LogTrace("Calibration") << "  Average rate = " << averageRate;

      for (int i_wire = firstWire; i_wire <= lastWire; ++i_wire) {
        // From definition of "noisy cell"
        int bin = i_wire - firstWire + 1;
        double channelRate = (*lHisto).second->GetBinContent(bin);
        double rateOffset = (useAbsoluteRate_) ? 0. : averageRate;
        if ((channelRate - rateOffset) > maximumNoiseRate_) {
          DTWireId wireID((*lHisto).first, i_wire);
          statusMap.setCellNoise(wireID, true);
          LogVerbatim("Calibration") << ">>> Channel noisy: " << wireID;
        }
      }
    }
  }
  LogVerbatim("Calibration") << "Writing noise map object to DB";
  string record = "DTStatusFlagRcd";
  DTCalibDBUtils::writeToDB<DTStatusFlag>(record, statusMap);
}

DTNoiseCalibration::~DTNoiseCalibration() { rootFile_->Close(); }

string DTNoiseCalibration::getChannelName(const DTWireId& wId) const {
  string channelName = "Wh" + std::to_string(wId.wheel()) + "_St" + std::to_string(wId.station()) + "_Sec" +
                       std::to_string(wId.sector()) + "_SL" + std::to_string(wId.superlayer()) + "_L" +
                       std::to_string(wId.layer()) + "_W" + std::to_string(wId.wire());
  return channelName;
}

string DTNoiseCalibration::getLayerName(const DTLayerId& lId) const {
  const DTSuperLayerId dtSLId = lId.superlayerId();
  const DTChamberId dtChId = dtSLId.chamberId();
  string layerName = "W" + std::to_string(dtChId.wheel()) + "_St" + std::to_string(dtChId.station()) + "_Sec" +
                     std::to_string(dtChId.sector()) + "_SL" + std::to_string(dtSLId.superlayer()) + "_L" +
                     std::to_string(lId.layer());

  return layerName;
}

string DTNoiseCalibration::getSuperLayerName(const DTSuperLayerId& dtSLId) const {
  const DTChamberId dtChId = dtSLId.chamberId();

  string superLayerName = "W" + std::to_string(dtChId.wheel()) + "_St" + std::to_string(dtChId.station()) + "_Sec" +
                          std::to_string(dtChId.sector()) + "_SL" + std::to_string(dtSLId.superlayer());

  return superLayerName;
}

string DTNoiseCalibration::getChamberName(const DTChamberId& dtChId) const {
  string chamberName = "W" + std::to_string(dtChId.wheel()) + "_St" + std::to_string(dtChId.station()) + "_Sec" +
                       std::to_string(dtChId.sector());

  return chamberName;
}
