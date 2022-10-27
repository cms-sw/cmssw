
/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/CalibMuon/interface/DTnoiseDBValidation.h"

// Framework
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// Noise record
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include "TFile.h"
#include "TH1F.h"
#include <cmath>
#include <cstdio>
#include <sstream>

using namespace edm;
using namespace std;

DTnoiseDBValidation::DTnoiseDBValidation(const ParameterSet &pset)
    : labelDBRef_(esConsumes(edm::ESInputTag("", pset.getParameter<string>("labelDBRef")))),
      labelDB_(esConsumes(edm::ESInputTag("", pset.getParameter<string>("labelDB")))),
      muonGeomToken_(esConsumes<edm::Transition::BeginRun>()) {
  LogVerbatim("NoiseDBValidation") << "[DTnoiseDBValidation] Constructor called!";

  // Get the DQM needed services
  usesResource("DQMStore");
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder("DT/DtCalib/NoiseDBValidation");

  diffTestName_ = "noiseDifferenceInRange";
  if (pset.exists("diffTestName"))
    diffTestName_ = pset.getParameter<string>("diffTestName");

  wheelTestName_ = "noiseWheelOccInRange";
  if (pset.exists("wheelTestName"))
    wheelTestName_ = pset.getParameter<string>("wheelTestName");

  stationTestName_ = "noiseStationOccInRange";
  if (pset.exists("stationTestName"))
    stationTestName_ = pset.getParameter<string>("stationTestName");

  sectorTestName_ = "noiseSectorOccInRange";
  if (pset.exists("sectorTestName"))
    sectorTestName_ = pset.getParameter<string>("sectorTestName");

  layerTestName_ = "noiseLayerOccInRange";
  if (pset.exists("layerTestName"))
    layerTestName_ = pset.getParameter<string>("layerTestName");

  outputMEsInRootFile_ = false;
  if (pset.exists("OutputFileName")) {
    outputMEsInRootFile_ = true;
    outputFileName_ = pset.getParameter<std::string>("OutputFileName");
  }
}

DTnoiseDBValidation::~DTnoiseDBValidation() {}

void DTnoiseDBValidation::beginRun(const edm::Run &run, const EventSetup &setup) {
  noiseRefMap_ = &setup.getData(labelDBRef_);

  noiseMap_ = &setup.getData(labelDB_);
  ;

  // Get the geometry
  dtGeom = &setup.getData(muonGeomToken_);

  LogVerbatim("NoiseDBValidation") << "[DTnoiseDBValidation] Parameters initialization";

  noisyCellsRef_ = 0;
  noisyCellsValid_ = 0;

  // Histo booking
  diffHisto_ =
      dbe_->book1D("noisyCellDiff", "percentual (wrt the previous db) total number of noisy cells", 1, 0.5, 1.5);
  diffHisto_->setBinLabel(1, "Diff");
  wheelHisto_ = dbe_->book1D("wheelOccupancy", "percentual noisy cells occupancy per wheel", 5, -2.5, 2.5);
  wheelHisto_->setBinLabel(1, "Wh-2");
  wheelHisto_->setBinLabel(2, "Wh-1");
  wheelHisto_->setBinLabel(3, "Wh0");
  wheelHisto_->setBinLabel(4, "Wh1");
  wheelHisto_->setBinLabel(5, "Wh2");
  stationHisto_ = dbe_->book1D("stationOccupancy", "percentual noisy cells occupancy per station", 4, 0.5, 4.5);
  stationHisto_->setBinLabel(1, "St1");
  stationHisto_->setBinLabel(2, "St2");
  stationHisto_->setBinLabel(3, "St3");
  stationHisto_->setBinLabel(4, "St4");
  sectorHisto_ = dbe_->book1D("sectorOccupancy", "percentual noisy cells occupancy per sector", 12, 0.5, 12.5);
  sectorHisto_->setBinLabel(1, "Sect1");
  sectorHisto_->setBinLabel(2, "Sect2");
  sectorHisto_->setBinLabel(3, "Sect3");
  sectorHisto_->setBinLabel(4, "Sect4");
  sectorHisto_->setBinLabel(5, "Sect5");
  sectorHisto_->setBinLabel(6, "Sect6");
  sectorHisto_->setBinLabel(7, "Sect7");
  sectorHisto_->setBinLabel(8, "Sect8");
  sectorHisto_->setBinLabel(9, "Sect9");
  sectorHisto_->setBinLabel(10, "Sect10");
  sectorHisto_->setBinLabel(11, "Sect11");
  sectorHisto_->setBinLabel(12, "Sect12");
  layerHisto_ = dbe_->book1D("layerOccupancy", "percentual noisy cells occupancy per layer", 3, 0.5, 3.5);
  layerHisto_->setBinLabel(1, "First 10 bins");
  layerHisto_->setBinLabel(2, "Middle bins");
  layerHisto_->setBinLabel(3, "Last 10 bins");

  // map initialization
  map<int, int> whMap;
  whMap.clear();
  map<int, int> stMap;
  stMap.clear();
  map<int, int> sectMap;
  sectMap.clear();
  map<int, int> layerMap;
  layerMap.clear();

  // Loop over reference DB entries
  for (DTStatusFlag::const_iterator noise = noiseRefMap_->begin(); noise != noiseRefMap_->end(); noise++) {
    DTWireId wireId((*noise).first.wheelId,
                    (*noise).first.stationId,
                    (*noise).first.sectorId,
                    (*noise).first.slId,
                    (*noise).first.layerId,
                    (*noise).first.cellId);
    LogVerbatim("NoiseDBValidation") << "Ref. noisy wire: " << wireId;
    ++noisyCellsRef_;
  }

  // Loop over validation DB entries
  for (DTStatusFlag::const_iterator noise = noiseMap_->begin(); noise != noiseMap_->end(); noise++) {
    DTWireId wireId((*noise).first.wheelId,
                    (*noise).first.stationId,
                    (*noise).first.sectorId,
                    (*noise).first.slId,
                    (*noise).first.layerId,
                    (*noise).first.cellId);
    LogVerbatim("NoiseDBValidation") << "Valid. noisy wire: " << wireId;
    ++noisyCellsValid_;

    whMap[(*noise).first.wheelId]++;
    stMap[(*noise).first.stationId]++;
    sectMap[(*noise).first.sectorId]++;

    const DTTopology &dtTopo = dtGeom->layer(wireId.layerId())->specificTopology();
    const int lastWire = dtTopo.lastChannel();
    if ((*noise).first.cellId <= 10)
      layerMap[1]++;
    if ((*noise).first.cellId > 10 && (*noise).first.cellId < (lastWire - 10))
      layerMap[2]++;
    if ((*noise).first.cellId >= (lastWire - 10))
      layerMap[3]++;

    const DTChamberId chId = wireId.layerId().superlayerId().chamberId();
    if (noiseHistoMap_.find(chId) == noiseHistoMap_.end())
      bookHisto(chId);
    int binNumber = 4 * (wireId.superLayer() - 1) + wireId.layer();
    noiseHistoMap_[chId]->Fill(wireId.wire(), binNumber);
  }

  // histo filling
  double scale = 1 / double(noisyCellsRef_);
  diffHisto_->Fill(1, abs(noisyCellsRef_ - noisyCellsValid_) * scale);

  scale = 1 / double(noisyCellsValid_);
  for (map<int, int>::const_iterator wheel = whMap.begin(); wheel != whMap.end(); wheel++) {
    wheelHisto_->Fill((*wheel).first, ((*wheel).second) * scale);
  }
  for (map<int, int>::const_iterator station = stMap.begin(); station != stMap.end(); station++) {
    stationHisto_->Fill((*station).first, ((*station).second) * scale);
  }
  for (map<int, int>::const_iterator sector = sectMap.begin(); sector != sectMap.end(); sector++) {
    sectorHisto_->Fill((*sector).first, ((*sector).second) * scale);
  }
  for (map<int, int>::const_iterator layer = layerMap.begin(); layer != layerMap.end(); layer++) {
    layerHisto_->Fill((*layer).first, ((*layer).second) * scale);
  }
}

void DTnoiseDBValidation::endRun(edm::Run const &run, edm::EventSetup const &setup) {
  // test on difference histo
  // string testCriterionName;
  // testCriterionName =
  // parameters.getUntrackedParameter<string>("diffTestName","noiseDifferenceInRange");
  const QReport *theDiffQReport = diffHisto_->getQReport(diffTestName_);
  if (theDiffQReport) {
    vector<dqm::me_util::Channel> badChannels = theDiffQReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); channel != badChannels.end();
         channel++) {
      LogWarning("NoiseDBValidation") << " Bad partial difference of noisy channels! Contents : "
                                      << (*channel).getContents();
    }
  }
  // testCriterionName =
  // parameters.getUntrackedParameter<string>("wheelTestName","noiseWheelOccInRange");
  const QReport *theDiffQReport2 = wheelHisto_->getQReport(wheelTestName_);
  if (theDiffQReport2) {
    vector<dqm::me_util::Channel> badChannels = theDiffQReport2->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); channel != badChannels.end();
         channel++) {
      int wheel = (*channel).getBin() - 3;
      LogWarning("NoiseDBValidation") << " Bad percentual occupancy for wheel : " << wheel
                                      << "  Contents : " << (*channel).getContents();
    }
  }
  // testCriterionName =
  // parameters.getUntrackedParameter<string>("stationTestName","noiseStationOccInRange");
  const QReport *theDiffQReport3 = stationHisto_->getQReport(stationTestName_);
  if (theDiffQReport3) {
    vector<dqm::me_util::Channel> badChannels = theDiffQReport3->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); channel != badChannels.end();
         channel++) {
      LogWarning("NoiseDBValidation") << " Bad percentual occupancy for station : " << (*channel).getBin()
                                      << "  Contents : " << (*channel).getContents();
    }
  }
  // testCriterionName =
  // parameters.getUntrackedParameter<string>("sectorTestName","noiseSectorOccInRange");
  const QReport *theDiffQReport4 = sectorHisto_->getQReport(sectorTestName_);
  if (theDiffQReport4) {
    vector<dqm::me_util::Channel> badChannels = theDiffQReport4->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); channel != badChannels.end();
         channel++) {
      LogWarning("NoiseDBValidation") << " Bad percentual occupancy for sector : " << (*channel).getBin()
                                      << "  Contents : " << (*channel).getContents();
    }
  }
  // testCriterionName =
  // parameters.getUntrackedParameter<string>("layerTestName","noiseLayerOccInRange");
  const QReport *theDiffQReport5 = layerHisto_->getQReport(layerTestName_);
  if (theDiffQReport5) {
    vector<dqm::me_util::Channel> badChannels = theDiffQReport5->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); channel != badChannels.end();
         channel++) {
      if ((*channel).getBin() == 1)
        LogWarning("NoiseDBValidation") << " Bad percentual occupancy for the first 10 wires! Contents : "
                                        << (*channel).getContents();
      if ((*channel).getBin() == 2)
        LogWarning("NoiseDBValidation") << " Bad percentual occupancy for the middle wires! Contents : "
                                        << (*channel).getContents();
      if ((*channel).getBin() == 3)
        LogWarning("NoiseDBValidation") << " Bad percentual occupancy for the last 10 wires! Contents : "
                                        << (*channel).getContents();
    }
  }
}

void DTnoiseDBValidation::endJob() {
  // Write the histos in a ROOT file
  if (outputMEsInRootFile_)
    dbe_->save(outputFileName_);
}

void DTnoiseDBValidation::bookHisto(const DTChamberId &chId) {
  stringstream histoName;
  histoName << "NoiseOccupancy"
            << "_W" << chId.wheel() << "_St" << chId.station() << "_Sec" << chId.sector();

  if (noiseHistoMap_.find(chId) == noiseHistoMap_.end()) {  // Redundant check
    // Get the chamber from the geometry
    int nWiresMax = 0;
    const DTChamber *dtchamber = dtGeom->chamber(chId);
    const vector<const DTSuperLayer *> &superlayers = dtchamber->superLayers();

    // Loop over layers and find the max # of wires
    for (vector<const DTSuperLayer *>::const_iterator sl = superlayers.begin(); sl != superlayers.end();
         ++sl) {  // loop over SLs
      vector<const DTLayer *> layers = (*sl)->layers();
      for (vector<const DTLayer *>::const_iterator lay = layers.begin(); lay != layers.end();
           ++lay) {  // loop over layers
        int nWires = (*lay)->specificTopology().channels();
        if (nWires > nWiresMax)
          nWiresMax = nWires;
      }
    }

    noiseHistoMap_[chId] = dbe_->book2D(histoName.str(), "Noise occupancy", nWiresMax, 1, (nWiresMax + 1), 12, 1, 13);
    for (int i_sl = 1; i_sl <= 3; ++i_sl) {
      for (int i_lay = 1; i_lay <= 4; ++i_lay) {
        int binNumber = 4 * (i_sl - 1) + i_lay;
        stringstream label;
        label << "SL" << i_sl << ": L" << i_lay;
        noiseHistoMap_[chId]->setBinLabel(binNumber, label.str(), 2);
      }
    }
  }
}
