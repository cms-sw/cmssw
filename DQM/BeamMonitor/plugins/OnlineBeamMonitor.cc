/*
 * \file OnlineBeamMonitor.cc
 * \author Lorenzo Uplegger/FNAL
 * modified by Simone Gennai INFN/Bicocca
 */

// system includes
#include <memory>
#include <numeric>

// user includes
#include "DQM/BeamMonitor/plugins/OnlineBeamMonitor.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
#include "RecoVertex/BeamSpotProducer/interface/PVFitter.h"

using namespace std;
using namespace edm;
using namespace reco;

//----------------------------------------------------------------------------------------------------------------------
OnlineBeamMonitor::OnlineBeamMonitor(const ParameterSet& ps)
    : monitorName_(ps.getUntrackedParameter<string>("MonitorName")),
      bsOnlineToken_(consumes(ps.getUntrackedParameter<edm::InputTag>("OnlineBeamSpotLabel"))),
      bsHLTToken_(esConsumes()),
      bsLegacyToken_(esConsumes()),
      numberOfValuesToSave_(0),
      appendRunTxt_(ps.getUntrackedParameter<bool>("AppendRunToFileName")),
      writeDIPTxt_(ps.getUntrackedParameter<bool>("WriteDIPAscii")),
      outputDIPTxt_(ps.getUntrackedParameter<std::string>("DIPFileName")),
      shouldReadEvent_(true) {
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
  histoByCategoryNames_.insert(pair<string, string>("lumi", "Lumibased BeamSpotOnline"));

  for (const auto& itV : varNamesV_) {
    for (const auto& itM : histoByCategoryNames_) {
      histosMap_[itV][itM.first][itM.second] = nullptr;
    }
  }
}

void OnlineBeamMonitor::fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
  edm::ParameterSetDescription ps;
  ps.addUntracked<std::string>("MonitorName", "YourSubsystemName");
  ps.addUntracked<edm::InputTag>("OnlineBeamSpotLabel", edm::InputTag("hltOnlineBeamSpot"));
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
                                  "BS Choice: +1=HLT / -1=Legacy / -10=Fake (fallback to PCL) / 0=No Transient ",
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

// Generic try-catch template
template <typename T, typename Func>
T tryCatch(Func f, T errorValue) {
  try {
    LogDebug("BeamSpotOnlineParameters") << "Trying function" << std::endl;
    return f();
  } catch (const std::exception& e) {
    LogDebug("BeamSpotOnlineParameters") << "Caught exception" << std::endl;
    print_error(e);
    return errorValue;
  }
}

// Enum the BS string parameters
enum BSparameters {
  startTime = 0,  // 0 additional std::string parameters
  endTime = 1,    // 1
  lumiRange = 2,  // 2
  events = 3,     // 3 additional int parameters
  maxPV = 4,      // 4
  nPV = 5,        // 5
  meanPV = 6,     // 6 additional float parameters
  meanErrPV = 7,  // 7
  rmsPV = 8,      // 8
  rmsErrPV = 9,   // 9
  END_OF_TYPES = 10,
};

// Unified functor
using BeamSpotFunctor =
    std::function<std::variant<std::string, int, float>(BSparameters, const BeamSpotOnlineObjects&)>;

BeamSpotFunctor beamSpotFunctor = [](BSparameters param,
                                     const BeamSpotOnlineObjects& payload) -> std::variant<std::string, int, float> {
  switch (param) {
    case BSparameters::startTime:
      return payload.startTime();
    case BSparameters::endTime:
      return payload.endTime();
    case BSparameters::lumiRange:
      return payload.lumiRange();
    case BSparameters::events:
      return payload.usedEvents();
    case BSparameters::maxPV:
      return payload.maxPVs();
    case BSparameters::nPV:
      return payload.numPVs();
    case BSparameters::meanPV:
      return payload.meanPV();
    case BSparameters::meanErrPV:
      return payload.meanErrorPV();
    case BSparameters::rmsPV:
      return payload.rmsPV();
    case BSparameters::rmsErrPV:
      return payload.rmsErrorPV();
    default:
      throw std::invalid_argument("Unknown BS parameter");
  }
};

//----------------------------------------------------------------------------------------------------------------------
std::shared_ptr<onlinebeammonitor::BeamSpotInfo> OnlineBeamMonitor::globalBeginLuminosityBlock(
    const LuminosityBlock& iLumi, const EventSetup& iSetup) const {
  // Always create a beamspot group for each lumi weather we have results or not! Each Beamspot will be of unknown type!
  auto beamSpotInfo = std::make_shared<onlinebeammonitor::BeamSpotInfo>();
  return beamSpotInfo;
}

void OnlineBeamMonitor::fetchBeamSpotInformation(const Event& iEvent, const EventSetup& iSetup) {
  auto const& iLumi = iEvent.getLuminosityBlock();
  auto beamSpotInfo = luminosityBlockCache(iLumi.index());
  //Read BeamSpot from DB
  ESHandle<BeamSpotOnlineObjects> bsHLTHandle;
  ESHandle<BeamSpotOnlineObjects> bsLegacyHandle;
  ESHandle<BeamSpotObjects> bsOnlineHandle;

  // Additional values for DIP publication
  std::string startTimeStamp_ = "0";
  std::string startTimeStampHLT_ = "0";
  std::string startTimeStampLegacy_ = "0";
  std::string stopTimeStamp_ = "0";
  std::string stopTimeStampHLT_ = "0";
  std::string stopTimeStampLegacy_ = "0";
  std::string lumiRange_ = "0 - 0";
  std::string lumiRangeHLT_ = "0 - 0";
  std::string lumiRangeLegacy_ = "0 - 0";
  int events_ = 0;
  int eventsHLT_ = 0;
  int eventsLegacy_ = 0;
  int maxPV_ = 0;
  int maxPVHLT_ = 0;
  int maxPVLegacy_ = 0;
  int nPV_ = 0;
  int nPVHLT_ = 0;
  int nPVLegacy_ = 0;
  float meanPV_ = 0.;
  float meanPVHLT_ = 0.;
  float meanPVLegacy_ = 0.;
  float meanErrPV_ = 0.;
  float meanErrPVHLT_ = 0.;
  float meanErrPVLegacy_ = 0.;
  float rmsPV_ = 0.;
  float rmsPVHLT_ = 0.;
  float rmsPVLegacy_ = 0.;
  float rmsErrPV_ = 0.;
  float rmsErrPVHLT_ = 0.;
  float rmsErrPVLegacy_ = 0.;

  if (auto bsHLTHandle = iSetup.getHandle(bsHLTToken_)) {
    auto const& spotDB = *bsHLTHandle;

    auto fetchValue = [&](BSparameters param, auto defaultValue) {
      return tryCatch([&]() { return std::get<decltype(defaultValue)>(beamSpotFunctor(param, spotDB)); }, defaultValue);
    };

    startTimeStampHLT_ = fetchValue(BSparameters::startTime, std::string("0"));
    stopTimeStampHLT_ = fetchValue(BSparameters::endTime, std::string("0"));
    lumiRangeHLT_ = fetchValue(BSparameters::lumiRange, std::string("0 - 0"));
    eventsHLT_ = fetchValue(BSparameters::events, -999);
    maxPVHLT_ = fetchValue(BSparameters::maxPV, -999);
    nPVHLT_ = fetchValue(BSparameters::nPV, -999);
    meanPVHLT_ = fetchValue(BSparameters::meanPV, -999.0f);
    meanErrPVHLT_ = fetchValue(BSparameters::meanErrPV, -999.0f);
    rmsPVHLT_ = fetchValue(BSparameters::rmsPV, -999.0f);
    rmsErrPVHLT_ = fetchValue(BSparameters::rmsErrPV, -999.0f);

    // translate from BeamSpotObjects to reco::BeamSpot
    BeamSpot::Point apoint(spotDB.x(), spotDB.y(), spotDB.z());

    BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < reco::BeamSpot::dimension; ++i) {
      for (int j = 0; j < reco::BeamSpot::dimension; ++j) {
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
    LogError("OnlineBeamMonitor") << "The database BeamSpot (hlt record) is not valid at lumi: "
                                  << iLumi.id().luminosityBlock();
  }

  if (auto bsLegacyHandle = iSetup.getHandle(bsLegacyToken_)) {
    auto const& spotDB = *bsLegacyHandle;

    auto fetchValue = [&](BSparameters param, auto defaultValue) {
      return tryCatch([&]() { return std::get<decltype(defaultValue)>(beamSpotFunctor(param, spotDB)); }, defaultValue);
    };

    startTimeStampLegacy_ = fetchValue(BSparameters::startTime, std::string("0"));
    stopTimeStampLegacy_ = fetchValue(BSparameters::endTime, std::string("0"));
    lumiRangeLegacy_ = fetchValue(BSparameters::lumiRange, std::string("0 - 0"));
    eventsLegacy_ = fetchValue(BSparameters::events, -999);
    maxPVLegacy_ = fetchValue(BSparameters::maxPV, -999);
    nPVLegacy_ = fetchValue(BSparameters::nPV, -999);
    meanPVLegacy_ = fetchValue(BSparameters::meanPV, -999.0f);
    meanErrPVLegacy_ = fetchValue(BSparameters::meanErrPV, -999.0f);
    rmsPVLegacy_ = fetchValue(BSparameters::rmsPV, -999.0f);
    rmsErrPVLegacy_ = fetchValue(BSparameters::rmsErrPV, -999.0f);

    // translate from BeamSpotObjects to reco::BeamSpot
    BeamSpot::Point apoint(spotDB.x(), spotDB.y(), spotDB.z());

    BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < reco::BeamSpot::dimension; ++i) {
      for (int j = 0; j < reco::BeamSpot::dimension; ++j) {
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
    LogError("OnlineBeamMonitor") << "The database BeamSpot (legacy record) is not valid at lumi: "
                                  << iLumi.id().luminosityBlock();
  }

  if (auto bsOnlineHandle = iEvent.getHandle(bsOnlineToken_)) {
    auto const& spotOnline = *bsOnlineHandle;

    beamSpotInfo->beamSpotsMap_["Online"] = spotOnline;

    if (writeDIPTxt_) {
      std::ofstream outFile;

      std::string tmpname = outputDIPTxt_;
      int frun = iLumi.getRun().run();

      char index[15];
      if (appendRunTxt_ && writeDIPTxt_) {
        sprintf(index, "%s%i", "_Run", frun);
        tmpname.insert(outputDIPTxt_.length() - 4, index);
      }

      if (beamSpotInfo->beamSpotsMap_.find("Online") != beamSpotInfo->beamSpotsMap_.end()) {
        if (beamSpotInfo->beamSpotsMap_.find("HLT") != beamSpotInfo->beamSpotsMap_.end() &&
            beamSpotInfo->beamSpotsMap_["Online"].x0() == beamSpotInfo->beamSpotsMap_["HLT"].x0()) {
          startTimeStamp_ = startTimeStampHLT_;
          stopTimeStamp_ = stopTimeStampHLT_;
          lumiRange_ = lumiRangeHLT_;
          events_ = eventsHLT_;
          maxPV_ = maxPVHLT_;
          nPV_ = nPVHLT_;
          meanPV_ = meanPVHLT_;
          meanErrPV_ = meanErrPVHLT_;
          rmsPV_ = rmsPVHLT_;
          rmsErrPV_ = rmsErrPVHLT_;
        } else if (beamSpotInfo->beamSpotsMap_.find("Legacy") != beamSpotInfo->beamSpotsMap_.end() &&
                   beamSpotInfo->beamSpotsMap_["Online"].x0() == beamSpotInfo->beamSpotsMap_["Legacy"].x0()) {
          startTimeStamp_ = startTimeStampLegacy_;
          stopTimeStamp_ = stopTimeStampLegacy_;
          lumiRange_ = lumiRangeLegacy_;
          events_ = eventsLegacy_;
          maxPV_ = maxPVLegacy_;
          nPV_ = nPVLegacy_;
          meanPV_ = meanPVLegacy_;
          meanErrPV_ = meanErrPVLegacy_;
          rmsPV_ = rmsPVLegacy_;
          rmsErrPV_ = rmsErrPVLegacy_;
        }
      }

      outFile.open(tmpname.c_str());

      // Write out file for DIP publication
      outFile << "Runnumber " << frun << std::endl;
      outFile << "BeginTimeOfFit " << startTimeStamp_ << " " << 0 << std::endl;
      outFile << "EndTimeOfFit " << stopTimeStamp_ << " " << 0 << std::endl;
      outFile << "LumiRange " << lumiRange_ << std::endl;
      outFile << "Type " << spotOnline.type() << std::endl;
      outFile << "X0 " << spotOnline.x0() << std::endl;
      outFile << "Y0 " << spotOnline.y0() << std::endl;
      outFile << "Z0 " << spotOnline.z0() << std::endl;
      outFile << "sigmaZ0 " << spotOnline.sigmaZ() << std::endl;
      outFile << "dxdz " << spotOnline.dxdz() << std::endl;
      outFile << "dydz " << spotOnline.dydz() << std::endl;
      outFile << "BeamWidthX " << spotOnline.BeamWidthX() << std::endl;
      outFile << "BeamWidthY " << spotOnline.BeamWidthY() << std::endl;
      for (int i = 0; i < reco::BeamSpot::dimension; ++i) {
        outFile << "Cov(" << i << ",j) ";
        for (int j = 0; j < reco::BeamSpot::dimension; ++j) {
          outFile << spotOnline.covariance(i, j) << " ";
        }
        outFile << std::endl;
      }
      outFile << "EmittanceX " << spotOnline.emittanceX() << std::endl;
      outFile << "EmittanceY " << spotOnline.emittanceY() << std::endl;
      outFile << "BetaStar " << spotOnline.betaStar() << std::endl;
      outFile << "events " << events_ << std::endl;
      outFile << "meanPV " << meanPV_ << std::endl;
      outFile << "meanErrPV " << meanErrPV_ << std::endl;
      outFile << "rmsPV " << rmsPV_ << std::endl;
      outFile << "rmsErrPV " << rmsErrPV_ << std::endl;
      outFile << "maxPV " << maxPV_ << std::endl;
      outFile << "nPV " << nPV_ << std::endl;

      outFile.close();
    }
    //LogInfo("OnlineBeamMonitor")
    //  << *spotOnline << std::endl;
  } else {
    LogError("OnlineBeamMonitor") << "The online BeamSpot collection is not valid at lumi: "
                                  << iLumi.id().luminosityBlock();
  }
}

//----------------------------------------------------------------------------------------------------------------------
void OnlineBeamMonitor::globalEndLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
  processedLumis_.push_back(iLumi.id().luminosityBlock());
  auto beamSpotInfo = luminosityBlockCache(iLumi.index());

  //Setting up the choice
  if (beamSpotInfo->beamSpotsMap_.find("Online") != beamSpotInfo->beamSpotsMap_.end()) {
    if (beamSpotInfo->beamSpotsMap_.find("HLT") != beamSpotInfo->beamSpotsMap_.end() &&
        beamSpotInfo->beamSpotsMap_["Online"].x0() == beamSpotInfo->beamSpotsMap_["HLT"].x0()) {
      bsChoice_->Fill(iLumi.id().luminosityBlock(), 1);
      bsChoice_->setBinError(iLumi.id().luminosityBlock(), 0.05);
    } else if (beamSpotInfo->beamSpotsMap_.find("Legacy") != beamSpotInfo->beamSpotsMap_.end() &&
               beamSpotInfo->beamSpotsMap_["Online"].x0() == beamSpotInfo->beamSpotsMap_["Legacy"].x0()) {
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
      } else if (itM.second == "Lumibased BeamSpotOnline") {
        if (resultsMap.find("Online") != resultsMap.end()) {
          histo->setBinContent(iLumi.id().luminosityBlock(), resultsMap["Online"].first);
          histo->setBinError(iLumi.id().luminosityBlock(), resultsMap["Online"].second);
        }
      } else {
        LogInfo("OnlineBeamMonitor") << "The histosMap_ have a histogram named " << itM.second
                                     << " that I can't recognize in this loop!";
      }
    }
  }
  shouldReadEvent_ = true;
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
                    itHHH.first == "Lumibased BeamSpotOnline") {
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
                itHHH.first == "Lumibased BeamSpotOnline") {
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

void OnlineBeamMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  if (shouldReadEvent_) {
    fetchBeamSpotInformation(iEvent, iSetup);
    shouldReadEvent_ = false;
  }
}

DEFINE_FWK_MODULE(OnlineBeamMonitor);
