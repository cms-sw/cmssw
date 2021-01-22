/*
 * \file OnlineBeamMonitor.cc
 * \author Lorenzo Uplegger/FNAL
 * modified by Simone Gennai INFN/Bicocca
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DQM/BeamMonitor/plugins/OnlineBeamMonitor.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
#include "RecoVertex/BeamSpotProducer/interface/PVFitter.h"
#include <memory>

#include <numeric>

using namespace std;
using namespace edm;
using namespace reco;

//----------------------------------------------------------------------------------------------------------------------
OnlineBeamMonitor::OnlineBeamMonitor(const ParameterSet& ps)
    : monitorName_(ps.getUntrackedParameter<string>("MonitorName")),
      bsTransientToken_(esConsumes<edm::Transition::BeginLuminosityBlock>()),
      bsHLTToken_(esConsumes<edm::Transition::BeginLuminosityBlock>()),
      bsLegacyToken_(esConsumes<edm::Transition::BeginLuminosityBlock>()),
      numberOfValuesToSave_(0) {
  if (!monitorName_.empty())
    monitorName_ = monitorName_ + "/";

  processedLumis_.clear();

  varNamesV_.push_back("x");
  varNamesV_.push_back("y");
  varNamesV_.push_back("z");
  varNamesV_.push_back("sigmaX");
  varNamesV_.push_back("sigmaY");
  varNamesV_.push_back("sigmaZ");

  //histoByCategoryNames_.insert(pair<string, string>("run", "Coordinate"));
  //histoByCategoryNames_.insert(pair<string, string>("run", "PrimaryVertex fit-DataBase"));
  //histoByCategoryNames_.insert(pair<string, string>("run", "PrimaryVertex fit-BeamFit"));
  //histoByCategoryNames_.insert(pair<string, string>("run", "PrimaryVertex fit-Scalers"));
  //histoByCategoryNames_.insert(pair<string, string>("run", "PrimaryVertex-DataBase"));
  //histoByCategoryNames_.insert(pair<string, string>("run", "PrimaryVertex-BeamFit"));
  //histoByCategoryNames_.insert(pair<string, string>("run", "PrimaryVertex-Scalers"));

  histoByCategoryNames_.insert(pair<string, string>("lumi", "Lumibased BeamSpotHLT"));
  histoByCategoryNames_.insert(pair<string, string>("lumi", "Lumibased BeamSpotLegacy"));
  histoByCategoryNames_.insert(pair<string, string>("lumi", "Lumibased BeamSpotTransient"));

  for (const auto& itV : varNamesV_) {
    for (const auto& itM : histoByCategoryNames_) {
      histosMap_[itV][itM.first][itM.second] = nullptr;
    }
  }
}

void OnlineBeamMonitor::fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
  edm::ParameterSetDescription ps;

  ps.addUntracked<std::string>("MonitorName", "YourSubsystemName");
  iDesc.addDefault(ps);
}

//----------------------------------------------------------------------------------------------------------------------
void OnlineBeamMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                       edm::Run const& iRun,
                                       edm::EventSetup const& iSetup) {
  string name;
  string title;
  int firstLumi = 1;
  int lastLumi = 3000;
  for (auto& itM : histosMap_) {
    //Making histos per Lumi
    // x,y,z,sigmaX,sigmaY,sigmaZ
    for (auto& itMM : itM.second) {
      if (itMM.first != "run") {
        for (auto& itMMM : itMM.second) {
          name = string("h") + itM.first + itMMM.first;
          title = itM.first + "_{0} " + itMMM.first;
          if (itMM.first == "lumi") {
            ibooker.setCurrentFolder(monitorName_ + "Debug");
            itMMM.second = ibooker.book1D(name, title, lastLumi - firstLumi + 1, firstLumi - 0.5, lastLumi + 0.5);
            itMMM.second->setEfficiencyFlag();
          } else {
            LogInfo("OnlineBeamMonitorClient") << "Unrecognized category " << itMM.first;
          }
          if (itMMM.second != nullptr) {
            if (itMMM.first.find('-') != string::npos) {
              itMMM.second->setAxisTitle(string("#Delta ") + itM.first + "_{0} (cm)", 2);
            } else {
              itMMM.second->setAxisTitle(itM.first + "_{0} (cm)", 2);
            }
            itMMM.second->setAxisTitle("Lumisection", 1);
          }
        }
      }
    }
  }

  // create and cd into new folder
  ibooker.setCurrentFolder(monitorName_ + "Validation");
  //Book histograms
  bsChoice_ = ibooker.book1D("bsChoice",
                             "Choice between HLT (+1) and Legacy (-1) BS",
                             lastLumi - firstLumi + 1,
                             firstLumi - 0.5,
                             lastLumi + 0.5);
  bsChoice_->setAxisTitle("Lumisection", 1);
  bsChoice_->setAxisTitle("Choice", 2);
}

//----------------------------------------------------------------------------------------------------------------------
std::shared_ptr<onlinebeammonitor::NoCache> OnlineBeamMonitor::globalBeginLuminosityBlock(
    const LuminosityBlock& iLumi, const EventSetup& iSetup) const {
  // Always create a beamspot group for each lumi weather we have results or not! Each Beamspot will be of unknown type!

  processedLumis_.push_back(iLumi.id().luminosityBlock());
  //Read BeamSpot from DB
  ESHandle<BeamSpotOnlineObjects> bsHLTHandle;
  ESHandle<BeamSpotOnlineObjects> bsLegacyHandle;
  ESHandle<BeamSpotObjects> bsTransientHandle;

  if (auto bsHLTHandle = iSetup.getHandle(bsHLTToken_)) {
    auto const& spotDB = *bsHLTHandle;

    // translate from BeamSpotObjects to reco::BeamSpot
    BeamSpot::Point apoint(spotDB.GetX(), spotDB.GetY(), spotDB.GetZ());

    BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        matrix(i, j) = spotDB.GetCovariance(i, j);
      }
    }

    beamSpotsMap_["HLT"] =
        BeamSpot(apoint, spotDB.GetSigmaZ(), spotDB.Getdxdz(), spotDB.Getdydz(), spotDB.GetBeamWidthX(), matrix);

    BeamSpot* aSpot = &(beamSpotsMap_["HLT"]);

    aSpot->setBeamWidthY(spotDB.GetBeamWidthY());
    aSpot->setEmittanceX(spotDB.GetEmittanceX());
    aSpot->setEmittanceY(spotDB.GetEmittanceY());
    aSpot->setbetaStar(spotDB.GetBetaStar());

    if (spotDB.GetBeamType() == 2) {
      aSpot->setType(reco::BeamSpot::Tracker);
    } else {
      aSpot->setType(reco::BeamSpot::Fake);
    }
    //LogInfo("OnlineBeamMonitor")
    //  << *aSpot << std::endl;
  } else {
    LogInfo("OnlineBeamMonitor") << "Database BeamSpot is not valid at lumi: " << iLumi.id().luminosityBlock();
  }
  if (auto bsLegacyHandle = iSetup.getHandle(bsLegacyToken_)) {
    auto const& spotDB = *bsLegacyHandle;
    // translate from BeamSpotObjects to reco::BeamSpot
    BeamSpot::Point apoint(spotDB.GetX(), spotDB.GetY(), spotDB.GetZ());

    BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        matrix(i, j) = spotDB.GetCovariance(i, j);
      }
    }

    beamSpotsMap_["Legacy"] =
        BeamSpot(apoint, spotDB.GetSigmaZ(), spotDB.Getdxdz(), spotDB.Getdydz(), spotDB.GetBeamWidthX(), matrix);

    BeamSpot* aSpot = &(beamSpotsMap_["Legacy"]);

    aSpot->setBeamWidthY(spotDB.GetBeamWidthY());
    aSpot->setEmittanceX(spotDB.GetEmittanceX());
    aSpot->setEmittanceY(spotDB.GetEmittanceY());
    aSpot->setbetaStar(spotDB.GetBetaStar());

    if (spotDB.GetBeamType() == 2) {
      aSpot->setType(reco::BeamSpot::Tracker);
    } else {
      aSpot->setType(reco::BeamSpot::Fake);
    }
    //LogInfo("OnlineBeamMonitor")
    //  << *aSpot << std::endl;
  } else {
    LogInfo("OnlineBeamMonitor") << "Database BeamSpot is not valid at lumi: " << iLumi.id().luminosityBlock();
  }
  if (auto bsTransientHandle = iSetup.getHandle(bsTransientToken_)) {
    auto const& spotDB = *bsTransientHandle;

    // translate from BeamSpotObjects to reco::BeamSpot
    BeamSpot::Point apoint(spotDB.GetX(), spotDB.GetY(), spotDB.GetZ());

    BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        matrix(i, j) = spotDB.GetCovariance(i, j);
      }
    }

    beamSpotsMap_["Transient"] =
        BeamSpot(apoint, spotDB.GetSigmaZ(), spotDB.Getdxdz(), spotDB.Getdydz(), spotDB.GetBeamWidthX(), matrix);

    BeamSpot* aSpot = &(beamSpotsMap_["Transient"]);

    aSpot->setBeamWidthY(spotDB.GetBeamWidthY());
    aSpot->setEmittanceX(spotDB.GetEmittanceX());
    aSpot->setEmittanceY(spotDB.GetEmittanceY());
    aSpot->setbetaStar(spotDB.GetBetaStar());

    if (spotDB.GetBeamType() == 2) {
      aSpot->setType(reco::BeamSpot::Tracker);
    } else {
      aSpot->setType(reco::BeamSpot::Fake);
    }
    //LogInfo("OnlineBeamMonitor")
    //  << *aSpot << std::endl;
  } else {
    LogInfo("OnlineBeamMonitor") << "Database BeamSpot is not valid at lumi: " << iLumi.id().luminosityBlock();
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------------------------------------
void OnlineBeamMonitor::globalEndLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
  //Setting up the choice
  if (beamSpotsMap_.find("Transient") != beamSpotsMap_.end()) {
    if (beamSpotsMap_.find("HLT") != beamSpotsMap_.end() &&
        beamSpotsMap_["Transient"].x0() == beamSpotsMap_["HLT"].x0()) {
      bsChoice_->setBinContent(iLumi.id().luminosityBlock(), 1);
      bsChoice_->setBinError(iLumi.id().luminosityBlock(), 0.05);
    } else if (beamSpotsMap_.find("Legacy") != beamSpotsMap_.end() &&
               beamSpotsMap_["Transient"].x0() == beamSpotsMap_["Legacy"].x0()) {
      bsChoice_->setBinContent(iLumi.id().luminosityBlock(), -1);
      bsChoice_->setBinError(iLumi.id().luminosityBlock(), 0.05);
    } else {
      bsChoice_->setBinContent(iLumi.id().luminosityBlock(), -10);
      bsChoice_->setBinError(iLumi.id().luminosityBlock(), 0.05);
    }
  } else {
    bsChoice_->setBinContent(iLumi.id().luminosityBlock(), 0);
    bsChoice_->setBinError(iLumi.id().luminosityBlock(), 0.05);
  }

  //    "PV,BF..."      Value,Error
  map<std::string, pair<double, double> > resultsMap;
  vector<pair<double, double> > vertexResults;
  MonitorElement* histo = nullptr;
  for (const auto& itV : varNamesV_) {
    resultsMap.clear();
    for (const auto& itBS : beamSpotsMap_) {
      if (itBS.second.type() == BeamSpot::Tracker) {
        if (itV == "x") {
          resultsMap[itBS.first] = pair<double, double>(itBS.second.x0(), itBS.second.x0Error());
        } else if (itV == "y") {
          resultsMap[itBS.first] = pair<double, double>(itBS.second.y0(), itBS.second.y0Error());
        } else if (itV == "z") {
          resultsMap[itBS.first] = pair<double, double>(itBS.second.z0(), itBS.second.z0Error());
        } else if (itV == "sigmaX") {
          resultsMap[itBS.first] = pair<double, double>(itBS.second.BeamWidthX(), itBS.second.BeamWidthXError());
        } else if (itV == "sigmaY") {
          resultsMap[itBS.first] = pair<double, double>(itBS.second.BeamWidthY(), itBS.second.BeamWidthYError());
        } else if (itV == "sigmaZ") {
          resultsMap[itBS.first] = pair<double, double>(itBS.second.sigmaZ(), itBS.second.sigmaZ0Error());
        } else {
          LogInfo("OnlineBeamMonitor") << "The histosMap_ has been built with the name " << itV
                                       << " that I can't recognize!";
        }
      }
    }

    for (const auto& itM : histoByCategoryNames_) {
      if ((histo = histosMap_[itV][itM.first][itM.second]) == nullptr)
        continue;
      if (itM.second == "Lumibased BeamSpotHLT") {
        if (resultsMap.find("HLT") != resultsMap.end()) {
          histo->setBinContent(iLumi.id().luminosityBlock(), resultsMap["HLT"].first);
          histo->setBinError(iLumi.id().luminosityBlock(), resultsMap["HLT"].second);
        }
      } else if (itM.second == "Lumibased BeamSpotLegacy") {
        if (resultsMap.find("Legacy") != resultsMap.end()) {
          histo->setBinContent(iLumi.id().luminosityBlock(), resultsMap["Legacy"].first);
          histo->setBinError(iLumi.id().luminosityBlock(), resultsMap["Legacy"].second);
        }
      } else if (itM.second == "Lumibased BeamSpotTransient") {
        if (resultsMap.find("Transient") != resultsMap.end()) {
          histo->setBinContent(iLumi.id().luminosityBlock(), resultsMap["Transient"].first);
          histo->setBinError(iLumi.id().luminosityBlock(), resultsMap["Transient"].second);
        }
      } else {
        LogInfo("OnlineBeamMonitor") << "The histosMap_ have a histogram named " << itM.second
                                     << " that I can't recognize in this loop!";
      }
    }
  }
}

void OnlineBeamMonitor::dqmEndRun(edm::Run const&, edm::EventSetup const&) {
  if (processedLumis_.empty()) {
    return;
  }

  const double bigNumber = 1000000.;
  std::sort(processedLumis_.begin(), processedLumis_.end());
  int firstLumi = *processedLumis_.begin();
  int lastLumi = *(--processedLumis_.end());
  bsChoice_->getTH1()->GetXaxis()->SetRangeUser(firstLumi - 0.5, lastLumi + 0.5);
  for (auto& itH : histosMap_) {
    for (auto& itHH : itH.second) {
      double min = bigNumber;
      double max = -bigNumber;
      if (itHH.first != "run") {
        for (auto& itHHH : itHH.second) {
          if (itHHH.second != nullptr) {
            for (int bin = 1; bin <= itHHH.second->getTH1()->GetNbinsX(); bin++) {
              if (itHHH.second->getTH1()->GetBinError(bin) != 0 || itHHH.second->getTH1()->GetBinContent(bin) != 0) {
                if (itHHH.first == "Lumibased BeamSpotHLT" || itHHH.first == "Lumibased BeamSpotLegacy" ||
                    itHHH.first == "Lumibased BeamSpotTransient") {
                  if (min > itHHH.second->getTH1()->GetBinContent(bin)) {
                    min = itHHH.second->getTH1()->GetBinContent(bin);
                  }
                  if (max < itHHH.second->getTH1()->GetBinContent(bin)) {
                    max = itHHH.second->getTH1()->GetBinContent(bin);
                  }
                } else {
                  LogInfo("OnlineBeamMonitorClient") << "The histosMap_ have a histogram named " << itHHH.first
                                                     << " that I can't recognize in this loop!";
                }
              }
            }
          }
        }
        for (auto& itHHH : itHH.second) {
          if (itHHH.second != nullptr) {
            if (itHHH.first == "Lumibased BeamSpotHLT" || itHHH.first == "Lumibased BeamSpotLegacy" ||
                itHHH.first == "Lumibased BeamSpotTransient") {
              if ((max == -bigNumber && min == bigNumber) || max - min == 0) {
                itHHH.second->getTH1()->SetMinimum(itHHH.second->getTH1()->GetMinimum() - 0.01);
                itHHH.second->getTH1()->SetMaximum(itHHH.second->getTH1()->GetMaximum() + 0.01);
              } else {
                itHHH.second->getTH1()->SetMinimum(min - 0.1 * (max - min));
                itHHH.second->getTH1()->SetMaximum(max + 0.1 * (max - min));
              }
            } else {
              LogInfo("OnlineBeamMonitorClient")
                  << "The histosMap_ have a histogram named " << itHHH.first << " that I can't recognize in this loop!";
            }
            itHHH.second->getTH1()->GetXaxis()->SetRangeUser(firstLumi - 0.5, lastLumi + 0.5);
          }
        }
      }
    }
  }
}
DEFINE_FWK_MODULE(OnlineBeamMonitor);
