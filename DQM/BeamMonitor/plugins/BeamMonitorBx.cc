/*
 * \file BeamMonitorBx.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 *
 */

#include "DQM/BeamMonitor/plugins/BeamMonitorBx.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/View.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <numeric>
#include <cmath>
#include <TMath.h>
#include <iostream>
#include <TStyle.h>

using namespace std;
using namespace edm;
using namespace reco;

void BeamMonitorBx::formatFitTime(char* ts, const time_t& t) {
#define CET (+1)
#define CEST (+2)

  tm* ptm;
  ptm = gmtime(&t);
  sprintf(ts,
          "%4d-%02d-%02d %02d:%02d:%02d",
          ptm->tm_year,
          ptm->tm_mon + 1,
          ptm->tm_mday,
          (ptm->tm_hour + CEST) % 24,
          ptm->tm_min,
          ptm->tm_sec);

#ifdef STRIP_TRAILING_BLANKS_IN_TIMEZONE
  unsigned int b = strlen(ts);
  while (ts[--b] == ' ') {
    ts[b] = 0;
  }
#endif
}

//
// constructors and destructor
//
BeamMonitorBx::BeamMonitorBx(const ParameterSet& ps) : countBx_(0), countEvt_(0), countLumi_(0), resetHistos_(false) {
  parameters_ = ps;
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName", "YourSubsystemName");
  bsSrc_ = parameters_.getUntrackedParameter<InputTag>("beamSpot");
  fitNLumi_ = parameters_.getUntrackedParameter<int>("fitEveryNLumi", -1);
  resetFitNLumi_ = parameters_.getUntrackedParameter<int>("resetEveryNLumi", -1);

  dbe_ = Service<DQMStore>().operator->();

  if (!monitorName_.empty())
    monitorName_ = monitorName_ + "/";

  theBeamFitter = new BeamFitter(parameters_, consumesCollector());
  theBeamFitter->resetTrkVector();
  theBeamFitter->resetLSRange();
  theBeamFitter->resetRefTime();
  theBeamFitter->resetPVFitter();

  if (fitNLumi_ <= 0)
    fitNLumi_ = 1;
  beginLumiOfBSFit_ = endLumiOfBSFit_ = 0;
  refBStime[0] = refBStime[1] = 0;
  lastlumi_ = 0;
  nextlumi_ = 0;
  firstlumi_ = 0;
  processed_ = false;
  countGoodFit_ = 0;
}

BeamMonitorBx::~BeamMonitorBx() { delete theBeamFitter; }

//--------------------------------------------------------
void BeamMonitorBx::beginJob() {
  varMap["x0_bx"] = "X_{0}";
  varMap["y0_bx"] = "Y_{0}";
  varMap["z0_bx"] = "Z_{0}";
  varMap["sigmaX_bx"] = "#sigma_{X}";
  varMap["sigmaY_bx"] = "#sigma_{Y}";
  varMap["sigmaZ_bx"] = "#sigma_{Z}";

  varMap1["x"] = "X_{0} [cm]";
  varMap1["y"] = "Y_{0} [cm]";
  varMap1["z"] = "Z_{0} [cm]";
  varMap1["sigmaX"] = "#sigma_{X} [cm]";
  varMap1["sigmaY"] = "#sigma_{Y} [cm]";
  varMap1["sigmaZ"] = "#sigma_{Z} [cm]";

  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_ + "FitBx");
  // Results of good fit:
  BookTables(1, varMap, "");
  //if (resetFitNLumi_ > 0) BookTables(1,varMap,"all");

  // create and cd into new folders
  for (std::map<std::string, std::string>::const_iterator varName = varMap1.begin(); varName != varMap1.end();
       ++varName) {
    string subDir_ = "FitBx";
    subDir_ += "/";
    subDir_ += "All_";
    subDir_ += varName->first;
    dbe_->setCurrentFolder(monitorName_ + subDir_);
  }
  dbe_->setCurrentFolder(monitorName_ + "FitBx/All_nPVs");
}

//--------------------------------------------------------
void BeamMonitorBx::beginRun(const edm::Run& r, const EventSetup& context) {
  ftimestamp = r.beginTime().value();
  tmpTime = ftimestamp >> 32;
  startTime = refTime = tmpTime;
}

//--------------------------------------------------------
void BeamMonitorBx::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  int nthlumi = lumiSeg.luminosityBlock();
  const edm::TimeValue_t fbegintimestamp = lumiSeg.beginTime().value();
  const std::time_t ftmptime = fbegintimestamp >> 32;

  if (countLumi_ == 0) {
    beginLumiOfBSFit_ = nthlumi;
    refBStime[0] = ftmptime;
  }
  if (beginLumiOfBSFit_ == 0)
    beginLumiOfBSFit_ = nextlumi_;

  if (nthlumi < nextlumi_)
    return;

  if (nthlumi > nextlumi_) {
    if (countLumi_ != 0 && processed_) {
      FitAndFill(lumiSeg, lastlumi_, nextlumi_, nthlumi);
    }
    nextlumi_ = nthlumi;
    edm::LogInfo("LS|BX|BeamMonitorBx") << "Next Lumi to Fit: " << nextlumi_ << endl;
    if (refBStime[0] == 0)
      refBStime[0] = ftmptime;
  }
  countLumi_++;
  if (processed_)
    processed_ = false;
  edm::LogInfo("LS|BX|BeamMonitorBx") << "Begin of Lumi: " << nthlumi << endl;
}

// ----------------------------------------------------------
void BeamMonitorBx::analyze(const Event& iEvent, const EventSetup& iSetup) {
  bool readEvent_ = true;
  const int nthlumi = iEvent.luminosityBlock();
  if (nthlumi < nextlumi_) {
    readEvent_ = false;
    edm::LogWarning("BX|BeamMonitorBx") << "Spilt event from previous lumi section!" << endl;
  }
  if (nthlumi > nextlumi_) {
    readEvent_ = false;
    edm::LogWarning("BX|BeamMonitorBx") << "Spilt event from next lumi section!!!" << endl;
  }

  if (readEvent_) {
    countEvt_++;
    theBeamFitter->readEvent(iEvent);
    processed_ = true;
  }
}

//--------------------------------------------------------
void BeamMonitorBx::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& iSetup) {
  int nthlumi = lumiSeg.id().luminosityBlock();
  edm::LogInfo("LS|BX|BeamMonitorBx") << "Lumi of the last event before endLuminosityBlock: " << nthlumi << endl;

  if (nthlumi < nextlumi_)
    return;
  const edm::TimeValue_t fendtimestamp = lumiSeg.endTime().value();
  const std::time_t fendtime = fendtimestamp >> 32;
  tmpTime = refBStime[1] = fendtime;
}

//--------------------------------------------------------
void BeamMonitorBx::BookTables(int nBx, map<string, string>& vMap, string suffix_) {
  // to rebin histograms when number of bx increases
  dbe_->cd(monitorName_ + "FitBx");

  for (std::map<std::string, std::string>::const_iterator varName = vMap.begin(); varName != vMap.end(); ++varName) {
    string tmpName = varName->first;
    if (!suffix_.empty()) {
      tmpName += "_";
      tmpName += suffix_;
    }

    hs[tmpName] = dbe_->book2D(tmpName, varName->second, 3, 0, 3, nBx, 0, nBx);
    hs[tmpName]->setBinLabel(1, "bx", 1);
    hs[tmpName]->setBinLabel(2, varName->second, 1);
    hs[tmpName]->setBinLabel(3, "#Delta " + varName->second, 1);
    for (int i = 0; i < nBx; i++) {
      hs[tmpName]->setBinLabel(i + 1, " ", 2);
    }
    hs[tmpName]->getTH1()->SetOption("text");
    hs[tmpName]->Reset();
  }
}

//--------------------------------------------------------
void BeamMonitorBx::BookTrendHistos(
    bool plotPV, int nBx, map<string, string>& vMap, string subDir_, const TString& prefix_, const TString& suffix_) {
  int nType_ = 2;
  if (plotPV)
    nType_ = 4;
  std::ostringstream ss;
  std::ostringstream ss1;
  ss << setfill('0') << setw(5) << nBx;
  ss1 << nBx;

  for (int i = 0; i < nType_; i++) {
    for (std::map<std::string, std::string>::const_iterator varName = vMap.begin(); varName != vMap.end(); ++varName) {
      string tmpDir_ = subDir_ + "/All_" + varName->first;
      dbe_->cd(monitorName_ + tmpDir_);
      TString histTitle(varName->first);
      TString tmpName;
      if (prefix_ != "")
        tmpName = prefix_ + "_" + varName->first;
      if (suffix_ != "")
        tmpName = tmpName + "_" + suffix_;
      tmpName = tmpName + "_" + ss.str();

      TString histName(tmpName);
      string ytitle(varName->second);
      string xtitle("");
      string options("E1");
      bool createHisto = true;
      switch (i) {
        case 1:  // BS vs time
          histName.Insert(histName.Index("_bx_", 4), "_time");
          xtitle = "Time [UTC]  [Bx# " + ss1.str() + "]";
          if (ytitle.find("sigma") == string::npos)
            histTitle += " coordinate of beam spot vs time (Fit)";
          else
            histTitle = histTitle.Insert(5, " ") + " of beam spot vs time (Fit)";
          break;
        case 2:  // PV +/- sigmaPV vs lumi
          if (ytitle.find("sigma") == string::npos) {
            histName.Insert(0, "PV");
            histName.Insert(histName.Index("_bx_", 4), "_lumi");
            histTitle.Insert(0, "Avg. ");
            histTitle += " position of primary vtx vs lumi";
            xtitle = "Lumisection  [Bx# " + ss1.str() + "]";
            ytitle.insert(0, "PV");
            ytitle += " #pm #sigma_{PV";
            ytitle += varName->first;
            ytitle += "} (cm)";
          } else
            createHisto = false;
          break;
        case 3:  // PV +/- sigmaPV vs time
          if (ytitle.find("sigma") == string::npos) {
            histName.Insert(0, "PV");
            histName.Insert(histName.Index("_bx_", 4), "_time");
            histTitle.Insert(0, "Avg. ");
            histTitle += " position of primary vtx vs time";
            xtitle = "Time [UTC]  [Bx# " + ss1.str() + "]";
            ytitle.insert(0, "PV");
            ytitle += " #pm #sigma_{PV";
            ytitle += varName->first;
            ytitle += "} (cm)";
          } else
            createHisto = false;
          break;
        default:  // BS vs lumi
          histName.Insert(histName.Index("_bx_", 4), "_lumi");
          xtitle = "Lumisection  [Bx# " + ss1.str() + "]";
          if (ytitle.find("sigma") == string::npos)
            histTitle += " coordinate of beam spot vs lumi (Fit)";
          else
            histTitle = histTitle.Insert(5, " ") + " of beam spot vs lumi (Fit)";
          break;
      }
      // check if already exist
      if (dbe_->get(monitorName_ + tmpDir_ + "/" + string(histName.Data())))
        continue;

      if (createHisto) {
        edm::LogInfo("BX|BeamMonitorBx") << "histName = " << histName << "; histTitle = " << histTitle << std::endl;
        hst[histName] = dbe_->book1D(histName, histTitle, 40, 0.5, 40.5);
        hst[histName]->getTH1()->SetCanExtend(TH1::kAllAxes);
        hst[histName]->setAxisTitle(xtitle, 1);
        hst[histName]->setAxisTitle(ytitle, 2);
        hst[histName]->getTH1()->SetOption("E1");
        if (histName.Contains("time")) {
          hst[histName]->getTH1()->SetBins(3600, 0.5, 3600 + 0.5);
          hst[histName]->setAxisTimeDisplay(1);
          hst[histName]->setAxisTimeFormat("%H:%M:%S", 1);

          char eventTime[64];
          formatFitTime(eventTime, startTime);
          TDatime da(eventTime);
          if (debug_) {
            edm::LogInfo("BX|BeamMonitorBx") << "TimeOffset = ";
            da.Print();
          }
          hst[histName]->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
        }
      }
    }  //End of variable loop
  }    // End of type loop (lumi, time)

  // num of PVs(#Bx) per LS
  dbe_->cd(monitorName_ + subDir_ + "/All_nPVs");
  TString histName = "Trending_nPVs_lumi_bx_" + ss.str();
  string xtitle = "Lumisection  [Bx# " + ss1.str() + "]";

  hst[histName] = dbe_->book1D(histName, "Number of Good Reconstructed Vertices", 40, 0.5, 40.5);
  hst[histName]->setAxisTitle(xtitle, 1);
  hst[histName]->getTH1()->SetCanExtend(TH1::kAllAxes);
  hst[histName]->getTH1()->SetOption("E1");
}

//--------------------------------------------------------
void BeamMonitorBx::FitAndFill(const LuminosityBlock& lumiSeg, int& lastlumi, int& nextlumi, int& nthlumi) {
  if (nthlumi <= nextlumi)
    return;

  int currentlumi = nextlumi;
  edm::LogInfo("LS|BX|BeamMonitorBx") << "Lumi of the current fit: " << currentlumi << endl;
  lastlumi = currentlumi;
  endLumiOfBSFit_ = currentlumi;

  edm::LogInfo("BX|BeamMonitorBx") << "Time lapsed = " << tmpTime - refTime << std::endl;

  if (resetHistos_) {
    edm::LogInfo("BX|BeamMonitorBx") << "Resetting Histograms" << endl;
    theBeamFitter->resetCutFlow();
    resetHistos_ = false;
  }

  if (fitNLumi_ > 0)
    if (currentlumi % fitNLumi_ != 0)
      return;

  std::pair<int, int> fitLS = theBeamFitter->getFitLSRange();
  edm::LogInfo("LS|BX|BeamMonitorBx") << " [Fitter] Do BeamSpot Fit for LS = " << fitLS.first << " to " << fitLS.second
                                      << endl;
  edm::LogInfo("LS|BX|BeamMonitorBx") << " [BX] Do BeamSpot Fit for LS = " << beginLumiOfBSFit_ << " to "
                                      << endLumiOfBSFit_ << endl;

  edm::LogInfo("BX|BeamMonitorBx") << " [BxDebugTime] refBStime[0] = " << refBStime[0]
                                   << "; address =  " << &refBStime[0] << std::endl;
  edm::LogInfo("BX|BeamMonitorBx") << " [BxDebugTime] refBStime[1] = " << refBStime[1]
                                   << "; address =  " << &refBStime[1] << std::endl;

  if (theBeamFitter->runPVandTrkFitter()) {
    countGoodFit_++;
    edm::LogInfo("BX|BeamMonitorBx") << "Number of good fit = " << countGoodFit_ << endl;
    BeamSpotMapBx bsmap = theBeamFitter->getBeamSpotMap();
    std::map<int, int> npvsmap = theBeamFitter->getNPVsperBX();
    edm::LogInfo("BX|BeamMonitorBx") << "Number of bx = " << bsmap.size() << endl;
    if (bsmap.empty())
      return;
    if (countBx_ < bsmap.size()) {
      countBx_ = bsmap.size();
      BookTables(countBx_, varMap, "");
      BookTables(countBx_, varMap, "all");
      for (BeamSpotMapBx::const_iterator abspot = bsmap.begin(); abspot != bsmap.end(); ++abspot) {
        int bx = abspot->first;
        BookTrendHistos(false, bx, varMap1, "FitBx", "Trending", "bx");
      }
    }

    std::pair<int, int> LSRange = theBeamFitter->getFitLSRange();
    char tmpTitle[50];
    sprintf(tmpTitle, "%s %i %s %i %s", " [cm] (LS: ", LSRange.first, " to ", LSRange.second, ")");
    for (std::map<std::string, std::string>::const_iterator varName = varMap.begin(); varName != varMap.end();
         ++varName) {
      hs[varName->first]->setTitle(varName->second + " " + tmpTitle);
      hs[varName->first]->Reset();
    }

    if (countGoodFit_ == 1)
      firstlumi_ = LSRange.first;

    if (resetFitNLumi_ > 0) {
      char tmpTitle1[60];
      if (countGoodFit_ > 1)
        snprintf(tmpTitle1,
                 sizeof(tmpTitle1),
                 "%s %i %s %i %s",
                 " [cm] (LS: ",
                 firstlumi_,
                 " to ",
                 LSRange.second,
                 ") [weighted average]");
      else
        snprintf(tmpTitle1, sizeof(tmpTitle1), "%s", "Need at least two fits to calculate weighted average");
      for (std::map<std::string, std::string>::const_iterator varName = varMap.begin(); varName != varMap.end();
           ++varName) {
        TString tmpName = varName->first + "_all";
        hs[tmpName]->setTitle(varName->second + " " + tmpTitle1);
        hs[tmpName]->Reset();
      }
    }

    int nthBin = countBx_;
    for (BeamSpotMapBx::const_iterator abspot = bsmap.begin(); abspot != bsmap.end(); ++abspot, nthBin--) {
      reco::BeamSpot bs = abspot->second;
      int bx = abspot->first;
      int nPVs = npvsmap.find(bx)->second;
      edm::LogInfo("BeamMonitorBx") << "Number of PVs of bx#[" << bx << "] = " << nPVs << endl;
      // Tables
      FillTables(bx, nthBin, varMap, bs, "");
      // Histograms
      FillTrendHistos(bx, nPVs, varMap1, bs, "Trending");
    }
    // Calculate weighted beam spots
    weight(fbspotMap, bsmap);
    // Fill the results
    nthBin = countBx_;
    if (resetFitNLumi_ > 0 && countGoodFit_ > 1) {
      for (BeamSpotMapBx::const_iterator abspot = fbspotMap.begin(); abspot != fbspotMap.end(); ++abspot, nthBin--) {
        reco::BeamSpot bs = abspot->second;
        int bx = abspot->first;
        FillTables(bx, nthBin, varMap, bs, "all");
      }
    }
  }
  //   else
  //     edm::LogInfo("BeamMonitorBx") << "Bad Fit!!!" << endl;

  if (resetFitNLumi_ > 0 && currentlumi % resetFitNLumi_ == 0) {
    edm::LogInfo("LS|BX|BeamMonitorBx") << "Reset track collection for beam fit!!!" << endl;
    resetHistos_ = true;
    theBeamFitter->resetTrkVector();
    theBeamFitter->resetLSRange();
    theBeamFitter->resetRefTime();
    theBeamFitter->resetPVFitter();
    beginLumiOfBSFit_ = 0;
    refBStime[0] = 0;
  }
}

//--------------------------------------------------------
void BeamMonitorBx::FillTables(int nthbx, int nthbin_, map<string, string>& vMap, reco::BeamSpot& bs_, string suffix_) {
  map<string, pair<double, double> > val_;
  val_["x0_bx"] = pair<double, double>(bs_.x0(), bs_.x0Error());
  val_["y0_bx"] = pair<double, double>(bs_.y0(), bs_.y0Error());
  val_["z0_bx"] = pair<double, double>(bs_.z0(), bs_.z0Error());
  val_["sigmaX_bx"] = pair<double, double>(bs_.BeamWidthX(), bs_.BeamWidthXError());
  val_["sigmaY_bx"] = pair<double, double>(bs_.BeamWidthY(), bs_.BeamWidthYError());
  val_["sigmaZ_bx"] = pair<double, double>(bs_.sigmaZ(), bs_.sigmaZ0Error());

  for (std::map<std::string, std::string>::const_iterator varName = vMap.begin(); varName != vMap.end(); ++varName) {
    TString tmpName = varName->first;
    if (!suffix_.empty())
      tmpName += TString("_" + suffix_);
    hs[tmpName]->setBinContent(1, nthbin_, nthbx);
    hs[tmpName]->setBinContent(2, nthbin_, val_[varName->first].first);
    hs[tmpName]->setBinContent(3, nthbin_, val_[varName->first].second);
  }
}

//--------------------------------------------------------
void BeamMonitorBx::FillTrendHistos(
    int nthBx, int nPVs_, map<string, string>& vMap, reco::BeamSpot& bs_, const TString& prefix_) {
  map<TString, pair<double, double> > val_;
  val_[TString(prefix_ + "_x")] = pair<double, double>(bs_.x0(), bs_.x0Error());
  val_[TString(prefix_ + "_y")] = pair<double, double>(bs_.y0(), bs_.y0Error());
  val_[TString(prefix_ + "_z")] = pair<double, double>(bs_.z0(), bs_.z0Error());
  val_[TString(prefix_ + "_sigmaX")] = pair<double, double>(bs_.BeamWidthX(), bs_.BeamWidthXError());
  val_[TString(prefix_ + "_sigmaY")] = pair<double, double>(bs_.BeamWidthY(), bs_.BeamWidthYError());
  val_[TString(prefix_ + "_sigmaZ")] = pair<double, double>(bs_.sigmaZ(), bs_.sigmaZ0Error());

  std::ostringstream ss;
  ss << setfill('0') << setw(5) << nthBx;
  int ntbin_ = tmpTime - startTime;
  for (map<TString, MonitorElement*>::iterator itHst = hst.begin(); itHst != hst.end(); ++itHst) {
    if (!(itHst->first.Contains(ss.str())))
      continue;
    if (itHst->first.Contains("nPVs"))
      continue;
    edm::LogInfo("BX|BeamMonitorBx") << "Filling histogram: " << itHst->first << endl;
    if (itHst->first.Contains("time")) {
      int idx = itHst->first.Index("_time", 5);
      itHst->second->setBinContent(ntbin_, val_[itHst->first(0, idx)].first);
      itHst->second->setBinError(ntbin_, val_[itHst->first(0, idx)].second);
    }
    if (itHst->first.Contains("lumi")) {
      int idx = itHst->first.Index("_lumi", 5);
      itHst->second->setBinContent(endLumiOfBSFit_, val_[itHst->first(0, idx)].first);
      itHst->second->setBinError(endLumiOfBSFit_, val_[itHst->first(0, idx)].second);
    }
  }
  TString histName = "Trending_nPVs_lumi_bx_" + ss.str();
  if (hst[histName])
    hst[histName]->setBinContent(endLumiOfBSFit_, nPVs_);
}

//--------------------------------------------------------------------------------------------------
void BeamMonitorBx::weight(BeamSpotMapBx& weightedMap_, const BeamSpotMapBx& newMap_) {
  for (BeamSpotMapBx::const_iterator it = newMap_.begin(); it != newMap_.end(); ++it) {
    if (weightedMap_.find(it->first) == weightedMap_.end() || (it->second.type() != 2)) {
      weightedMap_[it->first] = it->second;
      continue;
    }

    BeamSpot obs = weightedMap_[it->first];
    double val_[8] = {
        obs.x0(), obs.y0(), obs.z0(), obs.sigmaZ(), obs.dxdz(), obs.dydz(), obs.BeamWidthX(), obs.BeamWidthY()};
    double valErr_[8] = {obs.x0Error(),
                         obs.y0Error(),
                         obs.z0Error(),
                         obs.sigmaZ0Error(),
                         obs.dxdzError(),
                         obs.dydzError(),
                         obs.BeamWidthXError(),
                         obs.BeamWidthYError()};

    reco::BeamSpot::BeamType type = reco::BeamSpot::Unknown;
    weight(val_[0], valErr_[0], it->second.x0(), it->second.x0Error());
    weight(val_[1], valErr_[1], it->second.y0(), it->second.y0Error());
    weight(val_[2], valErr_[2], it->second.z0(), it->second.z0Error());
    weight(val_[3], valErr_[3], it->second.sigmaZ(), it->second.sigmaZ0Error());
    weight(val_[4], valErr_[4], it->second.dxdz(), it->second.dxdzError());
    weight(val_[5], valErr_[5], it->second.dydz(), it->second.dydzError());
    weight(val_[6], valErr_[6], it->second.BeamWidthX(), it->second.BeamWidthXError());
    weight(val_[7], valErr_[7], it->second.BeamWidthY(), it->second.BeamWidthYError());
    if (it->second.type() == reco::BeamSpot::Tracker) {
      type = reco::BeamSpot::Tracker;
    }

    reco::BeamSpot::Point bsPosition(val_[0], val_[1], val_[2]);
    reco::BeamSpot::CovarianceMatrix error;
    for (int i = 0; i < 7; ++i) {
      error(i, i) = valErr_[i] * valErr_[i];
    }
    reco::BeamSpot weightedBeamSpot(bsPosition, val_[3], val_[4], val_[5], val_[6], error, type);
    weightedBeamSpot.setBeamWidthY(val_[7]);
    LogInfo("BX|BeamMonitorBx") << weightedBeamSpot << endl;
    weightedMap_.erase(it->first);
    weightedMap_[it->first] = weightedBeamSpot;
  }
}

//--------------------------------------------------------------------------------------------------
void BeamMonitorBx::weight(double& mean, double& meanError, const double& val, const double& valError) {
  double tmpError = 0;
  if (meanError < 1e-8) {
    tmpError = 1 / (valError * valError);
    mean = val * tmpError;
  } else {
    tmpError = 1 / (meanError * meanError) + 1 / (valError * valError);
    mean = mean / (meanError * meanError) + val / (valError * valError);
  }
  mean = mean / tmpError;
  meanError = std::sqrt(1 / tmpError);
}

//--------------------------------------------------------
void BeamMonitorBx::endRun(const Run& r, const EventSetup& context) {}

DEFINE_FWK_MODULE(BeamMonitorBx);
