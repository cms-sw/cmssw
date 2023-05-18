/*
 * \file OnlineBeamMonitor.cc
 * \author Lorenzo Uplegger/FNAL
 * modified by Simone Gennai INFN/Bicocca
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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
      numberOfValuesToSave_(0),
      appendRunTxt_(ps.getUntrackedParameter<bool>("AppendRunToFileName")),
      writeDIPTxt_(ps.getUntrackedParameter<bool>("WriteDIPAscii")),
      outputDIPTxt_(ps.getUntrackedParameter<std::string>("DIPFileName")) {
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
  ps.addUntracked<bool>("AppendRunToFileName", false);
  ps.addUntracked<bool>("WriteDIPAscii", true);
  ps.addUntracked<std::string>("DIPFileName", "BeamFitResultsForDIP.txt");

  iDesc.addWithDefaultLabel(ps);
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
  bsChoice_ = ibooker.bookProfile("bsChoice",
                                  "BS Choice (+1): HLT - (-1): Legacy - (-10): Fake BS - (0): No Transient ",
                                  lastLumi - firstLumi + 1,
                                  firstLumi - 0.5,
                                  lastLumi + 0.5,
                                  100,
                                  -10,
                                  1,
                                  "");
  bsChoice_->setAxisTitle("Lumisection", 1);
  bsChoice_->setAxisTitle("Choice", 2);
}

//----------------------------------------------------------------------------------------------------------------------
// Handle exceptions for the schema evolution of the BeamSpotOnline CondFormat

// Slightly better error handler
static void print_error(const std::exception& e) { edm::LogError("BeamSpotOnlineParameters") << e.what() << '\n'; }

// Method to catch exceptions
template <typename T, class Except, class Func, class Response>
T try_(Func f, Response r) {
  try {
    LogDebug("BeamSpotOnlineParameters") << "I have tried" << std::endl;
    return f();
  } catch (Except& e) {
    LogDebug("BeamSpotOnlineParameters") << "I have caught!" << std::endl;
    r(e);
    return static_cast<T>("-999");
  }
}

// Enum the BS string parameters
enum BSparameters {
  startTime = 0,  // 0 additional std::string parameters
  endTime = 1,    // 1
  lumiRange = 2,  // 2
  END_OF_TYPES = 3,
};

// Functor
std::function<std::string(BSparameters, BeamSpotOnlineObjects)> myStringFunctor = [](BSparameters my_param,
                                                                                     BeamSpotOnlineObjects m_payload) {
  std::string ret("");
  switch (my_param) {
    case startTime:
      return m_payload.startTime();
    case endTime:
      return m_payload.endTime();
    case lumiRange:
      return m_payload.lumiRange();
    default:
      return ret;
  }
};

//----------------------------------------------------------------------------------------------------------------------
std::shared_ptr<onlinebeammonitor::BeamSpotInfo> OnlineBeamMonitor::globalBeginLuminosityBlock(
    const LuminosityBlock& iLumi, const EventSetup& iSetup) const {
  // Always create a beamspot group for each lumi weather we have results or not! Each Beamspot will be of unknown type!

  auto beamSpotInfo = std::make_shared<onlinebeammonitor::BeamSpotInfo>();

  //Read BeamSpot from DB
  ESHandle<BeamSpotOnlineObjects> bsHLTHandle;
  ESHandle<BeamSpotOnlineObjects> bsLegacyHandle;
  ESHandle<BeamSpotObjects> bsTransientHandle;
  //int lastLumiHLT_ = 0;
  //int lastLumiLegacy_ = 0;
  std::string startTimeStamp_ = "0";
  std::string startTimeStampHLT_ = "0";
  std::string startTimeStampLegacy_ = "0";
  std::string stopTimeStamp_ = "0";
  std::string stopTimeStampHLT_ = "0";
  std::string stopTimeStampLegacy_ = "0";
  std::string lumiRange_ = "0 - 0";
  std::string lumiRangeHLT_ = "0 - 0";
  std::string lumiRangeLegacy_ = "0 - 0";

  if (auto bsHLTHandle = iSetup.getHandle(bsHLTToken_)) {
    auto const& spotDB = *bsHLTHandle;

    //lastLumiHLT_ = spotDB.lastAnalyzedLumi();
    startTimeStampHLT_ =
        try_<std::string, std::out_of_range>(std::bind(myStringFunctor, BSparameters::startTime, spotDB), print_error);
    stopTimeStampHLT_ =
        try_<std::string, std::out_of_range>(std::bind(myStringFunctor, BSparameters::endTime, spotDB), print_error);
    lumiRangeHLT_ =
        try_<std::string, std::out_of_range>(std::bind(myStringFunctor, BSparameters::lumiRange, spotDB), print_error);

    // translate from BeamSpotObjects to reco::BeamSpot
    BeamSpot::Point apoint(spotDB.x(), spotDB.y(), spotDB.z());

    BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        matrix(i, j) = spotDB.covariance(i, j);
      }
    }

    beamSpotInfo->beamSpotsMap_["HLT"] =
        BeamSpot(apoint, spotDB.sigmaZ(), spotDB.dxdz(), spotDB.dydz(), spotDB.beamWidthX(), matrix);

    BeamSpot* aSpot = &(beamSpotInfo->beamSpotsMap_["HLT"]);

    aSpot->setBeamWidthY(spotDB.beamWidthY());
    aSpot->setEmittanceX(spotDB.emittanceX());
    aSpot->setEmittanceY(spotDB.emittanceY());
    aSpot->setbetaStar(spotDB.betaStar());

    if (spotDB.beamType() == 2) {
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
    BeamSpot::Point apoint(spotDB.x(), spotDB.y(), spotDB.z());

    //lastLumiLegacy_ = spotDB.lastAnalyzedLumi();
    startTimeStampLegacy_ =
        try_<std::string, std::out_of_range>(std::bind(myStringFunctor, BSparameters::startTime, spotDB), print_error);
    stopTimeStampLegacy_ =
        try_<std::string, std::out_of_range>(std::bind(myStringFunctor, BSparameters::endTime, spotDB), print_error);
    lumiRangeLegacy_ =
        try_<std::string, std::out_of_range>(std::bind(myStringFunctor, BSparameters::lumiRange, spotDB), print_error);

    BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        matrix(i, j) = spotDB.covariance(i, j);
      }
    }

    beamSpotInfo->beamSpotsMap_["Legacy"] =
        BeamSpot(apoint, spotDB.sigmaZ(), spotDB.dxdz(), spotDB.dydz(), spotDB.beamWidthX(), matrix);

    BeamSpot* aSpot = &(beamSpotInfo->beamSpotsMap_["Legacy"]);

    aSpot->setBeamWidthY(spotDB.beamWidthY());
    aSpot->setEmittanceX(spotDB.emittanceX());
    aSpot->setEmittanceY(spotDB.emittanceY());
    aSpot->setbetaStar(spotDB.betaStar());

    if (spotDB.beamType() == 2) {
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
    //std::cout << " from the DB " << spotDB << std::endl;

    // translate from BeamSpotObjects to reco::BeamSpot
    BeamSpot::Point apoint(spotDB.x(), spotDB.y(), spotDB.z());

    BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        matrix(i, j) = spotDB.covariance(i, j);
      }
    }

    beamSpotInfo->beamSpotsMap_["Transient"] =
        BeamSpot(apoint, spotDB.sigmaZ(), spotDB.dxdz(), spotDB.dydz(), spotDB.beamWidthX(), matrix);

    BeamSpot* aSpot = &(beamSpotInfo->beamSpotsMap_["Transient"]);

    aSpot->setBeamWidthY(spotDB.beamWidthY());
    aSpot->setEmittanceX(spotDB.emittanceX());
    aSpot->setEmittanceY(spotDB.emittanceY());
    aSpot->setbetaStar(spotDB.betaStar());
    if (spotDB.beamType() == 2) {
      aSpot->setType(reco::BeamSpot::Tracker);
    } else {
      aSpot->setType(reco::BeamSpot::Fake);
    }

    if (writeDIPTxt_) {
      std::ofstream outFile;

      std::string tmpname = outputDIPTxt_;
      int frun = iLumi.getRun().run();

      char index[15];
      if (appendRunTxt_ && writeDIPTxt_) {
        sprintf(index, "%s%i", "_Run", frun);
        tmpname.insert(outputDIPTxt_.length() - 4, index);
      }
      //int lastLumiAnalyzed_ = iLumi.id().luminosityBlock();

      if (beamSpotInfo->beamSpotsMap_.find("Transient") != beamSpotInfo->beamSpotsMap_.end()) {
        if (beamSpotInfo->beamSpotsMap_.find("HLT") != beamSpotInfo->beamSpotsMap_.end() &&
            beamSpotInfo->beamSpotsMap_["Transient"].x0() == beamSpotInfo->beamSpotsMap_["HLT"].x0()) {
          // lastLumiAnalyzed_ = lastLumiHLT_;
          startTimeStamp_ = startTimeStampHLT_;
          stopTimeStamp_ = stopTimeStampHLT_;
          lumiRange_ = lumiRangeHLT_;

        } else if (beamSpotInfo->beamSpotsMap_.find("Legacy") != beamSpotInfo->beamSpotsMap_.end() &&
                   beamSpotInfo->beamSpotsMap_["Transient"].x0() == beamSpotInfo->beamSpotsMap_["Legacy"].x0()) {
          //lastLumiAnalyzed_ = lastLumiLegacy_;
          startTimeStamp_ = startTimeStampLegacy_;
          stopTimeStamp_ = stopTimeStampLegacy_;
          lumiRange_ = lumiRangeLegacy_;
        }
      }

      outFile.open(tmpname.c_str());

      outFile << "Runnumber " << frun << " bx " << 0 << std::endl;
      outFile << "BeginTimeOfFit " << startTimeStamp_ << " " << 0 << std::endl;
      outFile << "EndTimeOfFit " << stopTimeStamp_ << " " << 0 << std::endl;
      //outFile << "LumiRange " << lumiRange_ << " - " << lastLumiAnalyzed_ << std::endl;
      outFile << "LumiRange " << lumiRange_ << std::endl;
      outFile << "Type " << aSpot->type() << std::endl;
      outFile << "X0 " << aSpot->x0() << std::endl;
      outFile << "Y0 " << aSpot->y0() << std::endl;
      outFile << "Z0 " << aSpot->z0() << std::endl;
      outFile << "sigmaZ0 " << aSpot->sigmaZ() << std::endl;
      outFile << "dxdz " << aSpot->dxdz() << std::endl;
      outFile << "dydz " << aSpot->dydz() << std::endl;
      outFile << "BeamWidthX " << aSpot->BeamWidthX() << std::endl;
      outFile << "BeamWidthY " << aSpot->BeamWidthY() << std::endl;
      for (int i = 0; i < 6; ++i) {
        outFile << "Cov(" << i << ",j) ";
        for (int j = 0; j < 7; ++j) {
          outFile << aSpot->covariance(i, j) << " ";
        }
        outFile << std::endl;
      }
      outFile << "Cov(6,j) 0 0 0 0 0 0 " << aSpot->covariance(6, 6) << std::endl;
      outFile << "EmittanceX " << aSpot->emittanceX() << std::endl;
      outFile << "EmittanceY " << aSpot->emittanceY() << std::endl;
      outFile << "BetaStar " << aSpot->betaStar() << std::endl;

      outFile.close();
    }
    //LogInfo("OnlineBeamMonitor")
    //  << *aSpot << std::endl;
  } else {
    LogInfo("OnlineBeamMonitor") << "Database BeamSpot is not valid at lumi: " << iLumi.id().luminosityBlock();
  }
  return beamSpotInfo;
}

//----------------------------------------------------------------------------------------------------------------------
void OnlineBeamMonitor::globalEndLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
  processedLumis_.push_back(iLumi.id().luminosityBlock());
  auto beamSpotInfo = luminosityBlockCache(iLumi.index());

  //Setting up the choice
  if (beamSpotInfo->beamSpotsMap_.find("Transient") != beamSpotInfo->beamSpotsMap_.end()) {
    if (beamSpotInfo->beamSpotsMap_.find("HLT") != beamSpotInfo->beamSpotsMap_.end() &&
        beamSpotInfo->beamSpotsMap_["Transient"].x0() == beamSpotInfo->beamSpotsMap_["HLT"].x0()) {
      bsChoice_->Fill(iLumi.id().luminosityBlock(), 1);
      bsChoice_->setBinError(iLumi.id().luminosityBlock(), 0.05);
    } else if (beamSpotInfo->beamSpotsMap_.find("Legacy") != beamSpotInfo->beamSpotsMap_.end() &&
               beamSpotInfo->beamSpotsMap_["Transient"].x0() == beamSpotInfo->beamSpotsMap_["Legacy"].x0()) {
      bsChoice_->Fill(iLumi.id().luminosityBlock(), -1);
      bsChoice_->setBinError(iLumi.id().luminosityBlock(), 0.05);
    } else {
      bsChoice_->Fill(iLumi.id().luminosityBlock(), -10);
      bsChoice_->setBinError(iLumi.id().luminosityBlock(), 0.05);
    }
  } else {
    bsChoice_->Fill(iLumi.id().luminosityBlock(), 0);
    bsChoice_->setBinError(iLumi.id().luminosityBlock(), 0.05);
  }

  //    "PV,BF..."      Value,Error
  map<std::string, pair<double, double> > resultsMap;
  vector<pair<double, double> > vertexResults;
  MonitorElement* histo = nullptr;
  for (const auto& itV : varNamesV_) {
    resultsMap.clear();
    for (const auto& itBS : beamSpotInfo->beamSpotsMap_) {
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
