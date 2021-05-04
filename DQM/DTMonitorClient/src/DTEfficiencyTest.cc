/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 */

#include <DQM/DTMonitorClient/src/DTEfficiencyTest.h>

// Framework
#include <FWCore/Framework/interface/EventSetup.h>

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cstdio>
#include <sstream>
#include <cmath>

using namespace edm;
using namespace std;

DTEfficiencyTest::DTEfficiencyTest(const edm::ParameterSet& ps) {
  edm::LogVerbatim("efficiency") << "[DTEfficiencyTest]: Constructor";

  parameters = ps;

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

  percentual = parameters.getUntrackedParameter<int>("BadSLpercentual", 10);
}

DTEfficiencyTest::~DTEfficiencyTest() {
  edm::LogVerbatim("efficiency") << "DTEfficiencyTest: analyzed " << nevents << " events";
}

void DTEfficiencyTest::beginRun(edm::Run const& run, edm::EventSetup const& context) {
  edm::LogVerbatim("efficiency") << "[DTEfficiencyTest]: Begin run";

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);
}

void DTEfficiencyTest::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                             DQMStore::IGetter& igetter,
                                             edm::LuminosityBlock const& lumiSeg,
                                             edm::EventSetup const& context) {
  for (map<int, MonitorElement*>::const_iterator histo = wheelHistos.begin(); histo != wheelHistos.end(); histo++) {
    (*histo).second->Reset();
  }

  for (map<int, MonitorElement*>::const_iterator histo = wheelUnassHistos.begin(); histo != wheelUnassHistos.end();
       histo++) {
    (*histo).second->Reset();
  }

  edm::LogVerbatim("efficiency") << "[DTEfficiencyTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if (nLumiSegs % prescaleFactor != 0)
    return;

  edm::LogVerbatim("efficiency") << "[DTEfficiencyTest]: " << nLumiSegs << " updates";

  vector<const DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  edm::LogVerbatim("efficiency") << "[DTEfficiencyTest]: Efficiency tests results";

  map<DTLayerId, vector<double> > LayerBadCells;
  LayerBadCells.clear();
  map<DTLayerId, vector<double> > LayerUnassBadCells;
  LayerUnassBadCells.clear();
  map<DTSuperLayerId, vector<double> > SuperLayerBadCells;
  SuperLayerBadCells.clear();
  map<DTSuperLayerId, vector<double> > SuperLayerUnassBadCells;
  SuperLayerUnassBadCells.clear();
  map<pair<int, int>, int> cmsHistos;
  cmsHistos.clear();
  map<pair<int, int>, bool> filled;
  for (int i = -2; i < 3; i++) {
    for (int j = 1; j < 15; j++) {
      filled[make_pair(i, j)] = false;
    }
  }
  map<pair<int, int>, int> cmsUnassHistos;
  cmsUnassHistos.clear();
  map<pair<int, int>, bool> UnassFilled;
  for (int i = -2; i < 3; i++) {
    for (int j = 1; j < 15; j++) {
      UnassFilled[make_pair(i, j)] = false;
    }
  }

  // Loop over the chambers
  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId chID = (*ch_it)->id();
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin();
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();

    // Loop over the SuperLayers
    for (; sl_it != sl_end; ++sl_it) {
      DTSuperLayerId slID = (*sl_it)->id();
      vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin();
      vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();

      // Loop over the layers
      for (; l_it != l_end; ++l_it) {
        DTLayerId lID = (*l_it)->id();

        stringstream wheel;
        wheel << chID.wheel();
        stringstream station;
        station << chID.station();
        stringstream sector;
        sector << chID.sector();
        stringstream superLayer;
        superLayer << slID.superlayer();
        stringstream layer;
        layer << lID.layer();

        string HistoName = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() + "_SL" +
                           superLayer.str() + "_L" + layer.str();

        // Get the ME produced by EfficiencyTask Source
        MonitorElement* occupancy_histo = igetter.get(getMEName("hEffOccupancy", lID));
        MonitorElement* unassOccupancy_histo = igetter.get(getMEName("hEffUnassOccupancy", lID));
        MonitorElement* recSegmOccupancy_histo = igetter.get(getMEName("hRecSegmOccupancy", lID));

        // ME -> TH1F
        if (occupancy_histo && unassOccupancy_histo && recSegmOccupancy_histo) {
          TH1F* occupancy_histo_root = occupancy_histo->getTH1F();
          TH1F* unassOccupancy_histo_root = unassOccupancy_histo->getTH1F();
          TH1F* recSegmOccupancy_histo_root = recSegmOccupancy_histo->getTH1F();

          const int firstWire = muonGeom->layer(lID)->specificTopology().firstChannel();
          const int lastWire = muonGeom->layer(lID)->specificTopology().lastChannel();

          // Loop over the TH1F bin and fill the ME to be used for the Quality Test
          for (int bin = firstWire; bin <= lastWire; bin++) {
            if ((recSegmOccupancy_histo_root->GetBinContent(bin)) != 0) {
              if (EfficiencyHistos.find(lID) == EfficiencyHistos.end())
                bookHistos(ibooker, lID, firstWire, lastWire);
              float efficiency =
                  occupancy_histo_root->GetBinContent(bin) / recSegmOccupancy_histo_root->GetBinContent(bin);
              float errorEff = sqrt(efficiency * (1 - efficiency) / recSegmOccupancy_histo_root->GetBinContent(bin));
              EfficiencyHistos.find(lID)->second->setBinContent(bin, efficiency);
              EfficiencyHistos.find(lID)->second->setBinError(bin, errorEff);

              if (UnassEfficiencyHistos.find(lID) == EfficiencyHistos.end())
                bookHistos(ibooker, lID, firstWire, lastWire);
              float unassEfficiency =
                  unassOccupancy_histo_root->GetBinContent(bin) / recSegmOccupancy_histo_root->GetBinContent(bin);
              float errorUnassEff =
                  sqrt(unassEfficiency * (1 - unassEfficiency) / recSegmOccupancy_histo_root->GetBinContent(bin));
              UnassEfficiencyHistos.find(lID)->second->setBinContent(bin, unassEfficiency);
              UnassEfficiencyHistos.find(lID)->second->setBinError(bin, errorUnassEff);
            }
          }
        }
      }  // loop on layers
    }    // loop on superlayers
  }      //loop on chambers

  // Efficiency test
  //cout<<"[DTEfficiencyTest]: Efficiency Tests results"<<endl;
  string EfficiencyCriterionName = parameters.getUntrackedParameter<string>("EfficiencyTestName", "EfficiencyInRange");
  for (map<DTLayerId, MonitorElement*>::const_iterator hEff = EfficiencyHistos.begin(); hEff != EfficiencyHistos.end();
       hEff++) {
    const QReport* theEfficiencyQReport = (*hEff).second->getQReport(EfficiencyCriterionName);
    double counter = 0;
    if (theEfficiencyQReport) {
      vector<dqm::me_util::Channel> badChannels = theEfficiencyQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); channel != badChannels.end();
           channel++) {
        edm::LogError("efficiency") << "LayerID : " << getMEName("hEffOccupancy", (*hEff).first)
                                    << " Bad efficiency channels: " << (*channel).getBin()
                                    << "  Contents : " << (*channel).getContents();
        counter++;
      }
      LayerBadCells[(*hEff).first].push_back(counter);
      LayerBadCells[(*hEff).first].push_back(muonGeom->layer((*hEff).first)->specificTopology().channels());
      // FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
      // edm::LogWarning ("efficiency") << "-------- "<<theEfficiencyQReport->getMessage()<<" ------- "<<theEfficiencyQReport->getStatus();
    }
  }

  // UnassEfficiency test
  //cout<<"[DTEfficiencyTest]: UnassEfficiency Tests results"<<endl;
  string UnassEfficiencyCriterionName =
      parameters.getUntrackedParameter<string>("UnassEfficiencyTestName", "UnassEfficiencyInRange");
  for (map<DTLayerId, MonitorElement*>::const_iterator hUnassEff = UnassEfficiencyHistos.begin();
       hUnassEff != UnassEfficiencyHistos.end();
       hUnassEff++) {
    const QReport* theUnassEfficiencyQReport = (*hUnassEff).second->getQReport(UnassEfficiencyCriterionName);
    double counter = 0;
    if (theUnassEfficiencyQReport) {
      vector<dqm::me_util::Channel> badChannels = theUnassEfficiencyQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); channel != badChannels.end();
           channel++) {
        edm::LogError("efficiency") << "Bad unassEfficiency channels: " << (*channel).getBin() << " "
                                    << (*channel).getContents();
        counter++;
      }
      LayerUnassBadCells[(*hUnassEff).first].push_back(counter);
      LayerUnassBadCells[(*hUnassEff).first].push_back(
          double(muonGeom->layer((*hUnassEff).first)->specificTopology().channels()));
      // FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
      // edm::LogWarning ("efficiency") << theUnassEfficiencyQReport->getMessage()<<" ------- "<<theUnassEfficiencyQReport->getStatus();
    }
  }

  vector<const DTChamber*>::const_iterator ch2_it = muonGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch2_end = muonGeom->chambers().end();
  for (; ch2_it != ch2_end; ++ch2_it) {
    vector<const DTSuperLayer*>::const_iterator sl2_it = (*ch2_it)->superLayers().begin();
    vector<const DTSuperLayer*>::const_iterator sl2_end = (*ch2_it)->superLayers().end();
    // Loop over the SLs
    for (; sl2_it != sl2_end; ++sl2_it) {
      DTSuperLayerId sl = (*sl2_it)->id();
      double superLayerBadC = 0;
      double superLayerTotC = 0;
      double superLayerUnassBadC = 0;
      double superLayerUnassTotC = 0;
      bool fill = false;
      vector<const DTLayer*>::const_iterator l2_it = (*sl2_it)->layers().begin();
      vector<const DTLayer*>::const_iterator l2_end = (*sl2_it)->layers().end();
      // Loop over the Ls
      for (; l2_it != l2_end; ++l2_it) {
        DTLayerId layerId = (*l2_it)->id();
        if (LayerBadCells.find(layerId) != LayerBadCells.end() &&
            LayerUnassBadCells.find(layerId) != LayerUnassBadCells.end()) {
          fill = true;
          superLayerBadC += LayerBadCells[layerId][0];
          superLayerTotC += LayerBadCells[layerId][1];
          superLayerUnassBadC += LayerUnassBadCells[layerId][0];
          superLayerUnassTotC += LayerUnassBadCells[layerId][1];
        }
      }
      if (fill) {
        SuperLayerBadCells[sl].push_back(superLayerBadC);
        SuperLayerBadCells[sl].push_back(superLayerTotC);
        SuperLayerUnassBadCells[sl].push_back(superLayerUnassBadC);
        SuperLayerUnassBadCells[sl].push_back(superLayerUnassTotC);
      }
    }
  }

  for (map<DTSuperLayerId, vector<double> >::const_iterator SLBCells = SuperLayerBadCells.begin();
       SLBCells != SuperLayerBadCells.end();
       SLBCells++) {
    if ((*SLBCells).second[0] / (*SLBCells).second[1] > double(percentual / 100)) {
      if (wheelHistos.find((*SLBCells).first.wheel()) == wheelHistos.end())
        bookHistos(ibooker, (*SLBCells).first.wheel());
      if (!((*SLBCells).first.station() == 4 && (*SLBCells).first.superlayer() == 3))
        wheelHistos[(*SLBCells).first.wheel()]->Fill(
            (*SLBCells).first.sector() - 1,
            ((*SLBCells).first.superlayer() - 1) + 3 * ((*SLBCells).first.station() - 1));
      else
        wheelHistos[(*SLBCells).first.wheel()]->Fill((*SLBCells).first.sector() - 1, 10);
      // fill the cms summary histo if the percentual of SL which have not passed the test
      // is more than a predefined treshold
      cmsHistos[make_pair((*SLBCells).first.wheel(), (*SLBCells).first.sector())]++;
      if (((*SLBCells).first.sector() < 13 &&
           double(cmsHistos[make_pair((*SLBCells).first.wheel(), (*SLBCells).first.sector())]) / 11 >
               double(percentual) / 100 &&
           filled[make_pair((*SLBCells).first.wheel(), (*SLBCells).first.sector())] == false) ||
          ((*SLBCells).first.sector() >= 13 &&
           double(cmsHistos[make_pair((*SLBCells).first.wheel(), (*SLBCells).first.sector())]) / 2 >
               double(percentual) / 100 &&
           filled[make_pair((*SLBCells).first.wheel(), (*SLBCells).first.sector())] == false)) {
        filled[make_pair((*SLBCells).first.wheel(), (*SLBCells).first.sector())] = true;
        wheelHistos[3]->Fill((*SLBCells).first.sector() - 1, (*SLBCells).first.wheel());
      }
    }
  }

  for (map<DTSuperLayerId, vector<double> >::const_iterator SLUBCells = SuperLayerUnassBadCells.begin();
       SLUBCells != SuperLayerUnassBadCells.end();
       SLUBCells++) {
    if ((*SLUBCells).second[0] / (*SLUBCells).second[1] > double(percentual / 100)) {
      if (wheelUnassHistos.find((*SLUBCells).first.wheel()) == wheelUnassHistos.end())
        bookHistos(ibooker, (*SLUBCells).first.wheel());
      if (!((*SLUBCells).first.station() == 4 && (*SLUBCells).first.superlayer() == 3))
        wheelUnassHistos[(*SLUBCells).first.wheel()]->Fill(
            (*SLUBCells).first.sector() - 1,
            ((*SLUBCells).first.superlayer() - 1) + 3 * ((*SLUBCells).first.station() - 1));
      else
        wheelUnassHistos[(*SLUBCells).first.wheel()]->Fill((*SLUBCells).first.sector() - 1, 10);
      // fill the cms summary histo if the percentual of SL which have not passed the test
      // is more than a predefined treshold
      cmsUnassHistos[make_pair((*SLUBCells).first.wheel(), (*SLUBCells).first.sector())]++;
      if (((*SLUBCells).first.sector() < 13 &&
           double(cmsUnassHistos[make_pair((*SLUBCells).first.wheel(), (*SLUBCells).first.sector())]) / 11 >
               double(percentual) / 100 &&
           UnassFilled[make_pair((*SLUBCells).first.wheel(), (*SLUBCells).first.sector())] == false) ||
          ((*SLUBCells).first.sector() >= 13 &&
           double(cmsUnassHistos[make_pair((*SLUBCells).first.wheel(), (*SLUBCells).first.sector())]) / 2 >
               double(percentual) / 100 &&
           UnassFilled[make_pair((*SLUBCells).first.wheel(), (*SLUBCells).first.sector())] == false)) {
        UnassFilled[make_pair((*SLUBCells).first.wheel(), (*SLUBCells).first.sector())] = true;
        wheelUnassHistos[3]->Fill((*SLUBCells).first.sector() - 1, (*SLUBCells).first.wheel());
      }
    }
  }
}

void DTEfficiencyTest::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  edm::LogVerbatim("efficiency") << "[DTEfficiencyTest] endjob called!";
}

string DTEfficiencyTest::getMEName(string histoTag, const DTLayerId& lID) {
  stringstream wheel;
  wheel << lID.superlayerId().wheel();
  stringstream station;
  station << lID.superlayerId().station();
  stringstream sector;
  sector << lID.superlayerId().sector();
  stringstream superLayer;
  superLayer << lID.superlayerId().superlayer();
  stringstream layer;
  layer << lID.layer();

  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderName = folderRoot + "DT/DTEfficiencyTask/Wheel" + wheel.str() + "/Station" + station.str() + "/Sector" +
                      sector.str() + "/SuperLayer" + superLayer.str() + "/";

  string histoname = folderName + histoTag + "_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() +
                     "_SL" + superLayer.str() + "_L" + layer.str();

  return histoname;
}

void DTEfficiencyTest::bookHistos(DQMStore::IBooker& ibooker, const DTLayerId& lId, int firstWire, int lastWire) {
  stringstream wheel;
  wheel << lId.superlayerId().wheel();
  stringstream station;
  station << lId.superlayerId().station();
  stringstream sector;
  sector << lId.superlayerId().sector();
  stringstream superLayer;
  superLayer << lId.superlayerId().superlayer();
  stringstream layer;
  layer << lId.layer();

  string HistoName =
      "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() + "_SL" + superLayer.str() + "_L" + layer.str();
  string EfficiencyHistoName = "Efficiency_" + HistoName;
  string UnassEfficiencyHistoName = "UnassEfficiency_" + HistoName;

  ibooker.setCurrentFolder("DT/Tests/DTEfficiency/Wheel" + wheel.str() + "/Station" + station.str() + "/Sector" +
                           sector.str());

  EfficiencyHistos[lId] = ibooker.book1D(EfficiencyHistoName.c_str(),
                                         EfficiencyHistoName.c_str(),
                                         lastWire - firstWire + 1,
                                         firstWire - 0.5,
                                         lastWire + 0.5);
  UnassEfficiencyHistos[lId] = ibooker.book1D(UnassEfficiencyHistoName.c_str(),
                                              UnassEfficiencyHistoName.c_str(),
                                              lastWire - firstWire + 1,
                                              firstWire - 0.5,
                                              lastWire + 0.5);
}

void DTEfficiencyTest::bookHistos(DQMStore::IBooker& ibooker, int wh) {
  ibooker.setCurrentFolder("DT/Tests/DTEfficiency/SummaryPlot");

  if (wheelHistos.find(3) == wheelHistos.end()) {
    string histoName = "ESummary_testFailedByAtLeastBadSL";
    wheelHistos[3] = ibooker.book2D(histoName.c_str(), histoName.c_str(), 14, 0, 14, 5, -2, 2);
    wheelHistos[3]->setBinLabel(1, "Sector1", 1);
    wheelHistos[3]->setBinLabel(1, "Sector1", 1);
    wheelHistos[3]->setBinLabel(2, "Sector2", 1);
    wheelHistos[3]->setBinLabel(3, "Sector3", 1);
    wheelHistos[3]->setBinLabel(4, "Sector4", 1);
    wheelHistos[3]->setBinLabel(5, "Sector5", 1);
    wheelHistos[3]->setBinLabel(6, "Sector6", 1);
    wheelHistos[3]->setBinLabel(7, "Sector7", 1);
    wheelHistos[3]->setBinLabel(8, "Sector8", 1);
    wheelHistos[3]->setBinLabel(9, "Sector9", 1);
    wheelHistos[3]->setBinLabel(10, "Sector10", 1);
    wheelHistos[3]->setBinLabel(11, "Sector11", 1);
    wheelHistos[3]->setBinLabel(12, "Sector12", 1);
    wheelHistos[3]->setBinLabel(13, "Sector13", 1);
    wheelHistos[3]->setBinLabel(14, "Sector14", 1);
    wheelHistos[3]->setBinLabel(1, "Wheel-2", 2);
    wheelHistos[3]->setBinLabel(2, "Wheel-1", 2);
    wheelHistos[3]->setBinLabel(3, "Wheel0", 2);
    wheelHistos[3]->setBinLabel(4, "Wheel+1", 2);
    wheelHistos[3]->setBinLabel(5, "Wheel+2", 2);
  }
  if (wheelUnassHistos.find(3) == wheelUnassHistos.end()) {
    string histoName = "UESummary_testFailedByAtLeastBadSL";
    wheelUnassHistos[3] = ibooker.book2D(histoName.c_str(), histoName.c_str(), 14, 0, 14, 5, -2, 2);
    wheelUnassHistos[3]->setBinLabel(1, "Sector1", 1);
    wheelUnassHistos[3]->setBinLabel(1, "Sector1", 1);
    wheelUnassHistos[3]->setBinLabel(2, "Sector2", 1);
    wheelUnassHistos[3]->setBinLabel(3, "Sector3", 1);
    wheelUnassHistos[3]->setBinLabel(4, "Sector4", 1);
    wheelUnassHistos[3]->setBinLabel(5, "Sector5", 1);
    wheelUnassHistos[3]->setBinLabel(6, "Sector6", 1);
    wheelUnassHistos[3]->setBinLabel(7, "Sector7", 1);
    wheelUnassHistos[3]->setBinLabel(8, "Sector8", 1);
    wheelUnassHistos[3]->setBinLabel(9, "Sector9", 1);
    wheelUnassHistos[3]->setBinLabel(10, "Sector10", 1);
    wheelUnassHistos[3]->setBinLabel(11, "Sector11", 1);
    wheelUnassHistos[3]->setBinLabel(12, "Sector12", 1);
    wheelUnassHistos[3]->setBinLabel(13, "Sector13", 1);
    wheelUnassHistos[3]->setBinLabel(14, "Sector14", 1);
    wheelUnassHistos[3]->setBinLabel(1, "Wheel-2", 2);
    wheelUnassHistos[3]->setBinLabel(2, "Wheel-1", 2);
    wheelUnassHistos[3]->setBinLabel(3, "Wheel0", 2);
    wheelUnassHistos[3]->setBinLabel(4, "Wheel+1", 2);
    wheelUnassHistos[3]->setBinLabel(5, "Wheel+2", 2);
  }

  stringstream wheel;
  wheel << wh;

  if (wheelHistos.find(wh) == wheelHistos.end()) {
    string histoName = "ESummary_testFailed_W" + wheel.str();
    wheelHistos[wh] = ibooker.book2D(histoName.c_str(), histoName.c_str(), 14, 0, 14, 11, 0, 11);
    wheelHistos[wh]->setBinLabel(1, "Sector1", 1);
    wheelHistos[wh]->setBinLabel(2, "Sector2", 1);
    wheelHistos[wh]->setBinLabel(3, "Sector3", 1);
    wheelHistos[wh]->setBinLabel(4, "Sector4", 1);
    wheelHistos[wh]->setBinLabel(5, "Sector5", 1);
    wheelHistos[wh]->setBinLabel(6, "Sector6", 1);
    wheelHistos[wh]->setBinLabel(7, "Sector7", 1);
    wheelHistos[wh]->setBinLabel(8, "Sector8", 1);
    wheelHistos[wh]->setBinLabel(9, "Sector9", 1);
    wheelHistos[wh]->setBinLabel(10, "Sector10", 1);
    wheelHistos[wh]->setBinLabel(11, "Sector11", 1);
    wheelHistos[wh]->setBinLabel(12, "Sector12", 1);
    wheelHistos[wh]->setBinLabel(13, "Sector13", 1);
    wheelHistos[wh]->setBinLabel(14, "Sector14", 1);
    wheelHistos[wh]->setBinLabel(1, "MB1_SL1", 2);
    wheelHistos[wh]->setBinLabel(2, "MB1_SL2", 2);
    wheelHistos[wh]->setBinLabel(3, "MB1_SL3", 2);
    wheelHistos[wh]->setBinLabel(4, "MB2_SL1", 2);
    wheelHistos[wh]->setBinLabel(5, "MB2_SL2", 2);
    wheelHistos[wh]->setBinLabel(6, "MB2_SL3", 2);
    wheelHistos[wh]->setBinLabel(7, "MB3_SL1", 2);
    wheelHistos[wh]->setBinLabel(8, "MB3_SL2", 2);
    wheelHistos[wh]->setBinLabel(9, "MB3_SL3", 2);
    wheelHistos[wh]->setBinLabel(10, "MB4_SL1", 2);
    wheelHistos[wh]->setBinLabel(11, "MB4_SL3", 2);
  }
  if (wheelUnassHistos.find(wh) == wheelUnassHistos.end()) {
    string histoName = "UESummary_testFailed_W" + wheel.str();
    wheelUnassHistos[wh] = ibooker.book2D(histoName.c_str(), histoName.c_str(), 14, 0, 14, 11, 0, 11);
    wheelUnassHistos[wh]->setBinLabel(1, "Sector1", 1);
    wheelUnassHistos[wh]->setBinLabel(2, "Sector2", 1);
    wheelUnassHistos[wh]->setBinLabel(3, "Sector3", 1);
    wheelUnassHistos[wh]->setBinLabel(4, "Sector4", 1);
    wheelUnassHistos[wh]->setBinLabel(5, "Sector5", 1);
    wheelUnassHistos[wh]->setBinLabel(6, "Sector6", 1);
    wheelUnassHistos[wh]->setBinLabel(7, "Sector7", 1);
    wheelUnassHistos[wh]->setBinLabel(8, "Sector8", 1);
    wheelUnassHistos[wh]->setBinLabel(9, "Sector9", 1);
    wheelUnassHistos[wh]->setBinLabel(10, "Sector10", 1);
    wheelUnassHistos[wh]->setBinLabel(11, "Sector11", 1);
    wheelUnassHistos[wh]->setBinLabel(12, "Sector12", 1);
    wheelUnassHistos[wh]->setBinLabel(13, "Sector13", 1);
    wheelUnassHistos[wh]->setBinLabel(14, "Sector14", 1);
    wheelUnassHistos[wh]->setBinLabel(1, "MB1_SL1", 2);
    wheelUnassHistos[wh]->setBinLabel(2, "MB1_SL2", 2);
    wheelUnassHistos[wh]->setBinLabel(3, "MB1_SL3", 2);
    wheelUnassHistos[wh]->setBinLabel(4, "MB2_SL1", 2);
    wheelUnassHistos[wh]->setBinLabel(5, "MB2_SL2", 2);
    wheelUnassHistos[wh]->setBinLabel(6, "MB2_SL3", 2);
    wheelUnassHistos[wh]->setBinLabel(7, "MB3_SL1", 2);
    wheelUnassHistos[wh]->setBinLabel(8, "MB3_SL2", 2);
    wheelUnassHistos[wh]->setBinLabel(9, "MB3_SL3", 2);
    wheelUnassHistos[wh]->setBinLabel(10, "MB4_SL1", 2);
    wheelUnassHistos[wh]->setBinLabel(11, "MB4_SL3", 2);
  }
}
