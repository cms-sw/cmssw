/*
 * \file BeamMonitorBx.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 * $Date: 2010/06/04 23:17:26 $
 * $Revision: 1.3 $
 *
 */

#include "DQM/BeamMonitor/plugins/BeamMonitorBx.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/View.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <numeric>
#include <math.h>
#include <TMath.h>
#include <iostream>
#include <TStyle.h>

using namespace std;
using namespace edm;

//
// constructors and destructor
//
BeamMonitorBx::BeamMonitorBx( const ParameterSet& ps ) :
  countBx_(0),countEvt_(0),countLumi_(0),resetHistos_(false) {

  parameters_     = ps;
  monitorName_    = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  bsSrc_          = parameters_.getUntrackedParameter<InputTag>("beamSpot");
  fitNLumi_       = parameters_.getUntrackedParameter<int>("fitEveryNLumi",-1);
  resetFitNLumi_  = parameters_.getUntrackedParameter<int>("resetEveryNLumi",-1);

  dbe_            = Service<DQMStore>().operator->();
  
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  
  theBeamFitter = new BeamFitter(parameters_);
  theBeamFitter->resetTrkVector();
  theBeamFitter->resetLSRange();
  theBeamFitter->resetRefTime();
  theBeamFitter->resetPVFitter();

  if (fitNLumi_ <= 0) fitNLumi_ = 1;
  beginLumiOfBSFit_ = endLumiOfBSFit_ = 0;
  refBStime[0] = refBStime[1] = 0;
  lastlumi_ = 0;
  nextlumi_ = 0;
  processed_ = false;
}


BeamMonitorBx::~BeamMonitorBx() {
  delete theBeamFitter;
}


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

  TDatime *da = new TDatime();
  gStyle->SetTimeOffset(da->Convert(kTRUE));

  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"FitBx");
  // Results of good fit:
  BookHistos(1,varMap);

  // create and cd into new folders
  for (std::map<std::string,std::string>::const_iterator varName = varMap1.begin();
       varName != varMap1.end(); ++varName) {
    string subDir_ = "FitBx";
    subDir_ += "/";
    subDir_ += "All_";
    subDir_ += (*varName).first;
    dbe_->setCurrentFolder(monitorName_+subDir_);
  }
}

//--------------------------------------------------------
void BeamMonitorBx::beginRun(const edm::Run& r, const EventSetup& context) {

  ftimestamp = r.beginTime().value();
  tmpTime = ftimestamp >> 32;
  startTime = refTime =  tmpTime;

}

//--------------------------------------------------------
void BeamMonitorBx::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
					 const EventSetup& context) {
  int nthlumi = lumiSeg.luminosityBlock();
  const edm::TimeValue_t fbegintimestamp = lumiSeg.beginTime().value();
  const std::time_t ftmptime = fbegintimestamp >> 32;

  if (countLumi_ == 0) {
    beginLumiOfBSFit_ = nthlumi;
    refBStime[0] = ftmptime;
  }
  if (beginLumiOfBSFit_ == 0) beginLumiOfBSFit_ = nextlumi_;

  if (nthlumi < nextlumi_) return;

  if (nthlumi > nextlumi_) {
    if (countLumi_ != 0 && processed_) {
      FitAndFill(lumiSeg,lastlumi_,nextlumi_,nthlumi);
    }
    nextlumi_ = nthlumi;
    edm::LogInfo("LS|BX|BeamMonitorBx") << "Next Lumi to Fit: " << nextlumi_ << endl;
    if (refBStime[0] == 0) refBStime[0] = ftmptime;
  }
  countLumi_++;
  if (processed_) processed_ = false;
  edm::LogInfo("LS|BX|BeamMonitorBx") << "Begin of Lumi: " << nthlumi << endl;
}

// ----------------------------------------------------------
void BeamMonitorBx::analyze(const Event& iEvent,
			    const EventSetup& iSetup ) {
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
void BeamMonitorBx::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				       const EventSetup& iSetup) {
  int nthlumi = lumiSeg.id().luminosityBlock();
  edm::LogInfo("LS|BX|BeamMonitorBx") << "Lumi of the last event before endLuminosityBlock: " << nthlumi << endl;

  if (nthlumi < nextlumi_) return;
  const edm::TimeValue_t fendtimestamp = lumiSeg.endTime().value();
  const std::time_t fendtime = fendtimestamp >> 32;
  tmpTime = refBStime[1] = fendtime;
}


//--------------------------------------------------------
void BeamMonitorBx::BookHistos(int nBx, map<string,string> & vMap) {
  // to rebin histograms when number of bx increases
  // create and cd into new folder
  dbe_->cd(monitorName_+"FitBx");

  for (std::map<std::string,std::string>::const_iterator varName = vMap.begin();
       varName != vMap.end(); ++varName) {
    string tmpName = varName->first;
    if (dbe_->get(monitorName_+"FitBx/"+tmpName)) {
      edm::LogInfo("BX|BeamMonitorBx") << "Rebinning " << tmpName << endl;
      dbe_->removeElement(tmpName);
    }

    hs[tmpName]=dbe_->book2D(tmpName,varName->second,3,0,3,nBx,0,nBx);
    hs[tmpName]->setBinLabel(1,"bx",1);
    hs[tmpName]->setBinLabel(2,varName->second,1);
    hs[tmpName]->setBinLabel(3,"#Delta "+varName->second,1);
    for (int i=0;i<nBx;i++) {
      hs[tmpName]->setBinLabel(i+1," ",2);
    }
    hs[tmpName]->getTH1()->SetOption("text");
    hs[tmpName]->Reset();
  }
}

//--------------------------------------------------------
void BeamMonitorBx::BookTrendHistos(bool plotPV,int nBx,map<string,string> & vMap,
				    string subDir_, TString prefix_, TString suffix_) {
  int nType_ = 2;
  if (plotPV) nType_ = 4;
  for (int i = 0; i < nType_; i++) {
    for (std::map<std::string,std::string>::const_iterator varName = vMap.begin();
	 varName != vMap.end(); ++varName) {
      string tmpDir_ = subDir_ + "/All_" + (*varName).first;
      dbe_->cd(monitorName_+tmpDir_);
      TString histTitle((*varName).first);
      string tmpName;
      if (prefix_ != "") tmpName = prefix_ + "_" + (*varName).first;
      if (suffix_ != "") tmpName = tmpName + "_" + suffix_;
      std::ostringstream ss;
      std::ostringstream ss1;
      ss << setfill ('0') << setw (5) << nBx;
      ss1 << nBx;
      tmpName = tmpName + "_" + ss.str();

      TString histName(tmpName);
      string ytitle((*varName).second);
      string xtitle("");
      string options("E1");
      bool createHisto = true;
      switch(i) {
      case 1: // BS vs time
	histName.Insert(histName.Index("_bx_",4),"_time");
	xtitle = "Time [UTC]  [Bx# " + ss1.str() + "]";
	if (ytitle.find("sigma") == string::npos)
	  histTitle += " coordinate of beam spot vs time (Fit)";
	else
	  histTitle = histTitle.Insert(5," ") + " of beam spot vs time (Fit)";
	break;
      case 2: // PV +/- sigmaPV vs lumi
	if (ytitle.find("sigma") == string::npos) {
	  histName.Insert(0,"PV");
	  histName.Insert(histName.Index("_bx_",4),"_lumi");
	  histTitle.Insert(0,"Avg. ");
	  histTitle += " position of primary vtx vs lumi";
	  xtitle = "Lumisection  [Bx# " + ss1.str() + "]";
	  ytitle.insert(0,"PV");
	  ytitle += " #pm #sigma_{PV";
	  ytitle += (*varName).first;
	  ytitle += "} (cm)";
	}
	else createHisto = false;
	break;
      case 3: // PV +/- sigmaPV vs time
	if (ytitle.find("sigma") == string::npos) {
	  histName.Insert(0,"PV");
	  histName.Insert(histName.Index("_bx_",4),"_time");
	  histTitle.Insert(0,"Avg. ");
	  histTitle += " position of primary vtx vs time";
	  xtitle = "Time [UTC]  [Bx# " + ss1.str() + "]";
	  ytitle.insert(0,"PV");
	  ytitle += " #pm #sigma_{PV";
	  ytitle += (*varName).first;
	  ytitle += "} (cm)";
	}
	else createHisto = false;
	break;
      default: // BS vs lumi
	histName.Insert(histName.Index("_bx_",4),"_lumi");
	xtitle = "Lumisection  [Bx# " + ss1.str() + "]";
	if (ytitle.find("sigma") == string::npos)
	  histTitle += " coordinate of beam spot vs lumi (Fit)";
	else
	  histTitle = histTitle.Insert(5," ") + " of beam spot vs lumi (Fit)";
	break;
      }
      // check if already exist
      if (dbe_->get(monitorName_+tmpDir_+"/"+string(histName))) continue;

      if (createHisto) {
	edm::LogInfo("BX|BeamMonitorBx") << "histName = " << histName << "; histTitle = " << histTitle << std::endl;
	hst[histName] = dbe_->book1D(histName,histTitle,40,0.5,40.5);
	hst[histName]->getTH1()->SetBit(TH1::kCanRebin);
	hst[histName]->setAxisTitle(xtitle,1);
	hst[histName]->setAxisTitle(ytitle,2);
	hst[histName]->getTH1()->SetOption("E1");
	if (histName.Contains("time")) {
	  hst[histName]->getTH1()->SetBins(3600,0.5,3600+0.5);
	  hst[histName]->setAxisTimeDisplay(1);
	  hst[histName]->setAxisTimeFormat("%H:%M:%S",1);
	}
      }
    }//End of variable loop
  }// End of type loop (lumi, time)
}

//--------------------------------------------------------
void BeamMonitorBx::FillTrendHistos(int nthBx, map<string,string> & vMap, 
				    reco::BeamSpot & bs_) {
  double val_[6] = {bs_.x0(),bs_.y0(),bs_.z0(),
		    bs_.BeamWidthX(),bs_.BeamWidthY(),bs_.sigmaZ()};
  double valErr_[6] = {bs_.x0Error(),bs_.y0Error(),bs_.z0Error(),
		       bs_.BeamWidthXError(),bs_.BeamWidthYError(),
		       bs_.sigmaZ0Error()};

  std::ostringstream ss;
  ss << setfill ('0') << setw (5) << nthBx;
  int ntbin_ = tmpTime - startTime;
  for (map<TString,MonitorElement*>::iterator itHst = hst.begin();
       itHst != hst.end(); ++itHst) {
    if (!((*itHst).first.Contains(ss.str()))) continue;
    int ic = 0;
    for (std::map<std::string,std::string>::const_iterator varName = vMap.begin();
	 varName != vMap.end(); ++varName, ++ic) {
      edm::LogInfo("BX|BeamMonitorBx") << "Filling " << (*itHst).first << endl;
      if ((*itHst).first.Contains("time")) {
	(*itHst).second->setBinContent(ntbin_,val_[ic]);
	(*itHst).second->setBinError(ntbin_,valErr_[ic]);
      }
      if ((*itHst).first.Contains("lumi")) {
	(*itHst).second->setBinContent(endLumiOfBSFit_,val_[ic]);
	(*itHst).second->setBinError(endLumiOfBSFit_,valErr_[ic]);
      }
    }
  }
}

//--------------------------------------------------------
void BeamMonitorBx::FitAndFill(const LuminosityBlock& lumiSeg,
			       int &lastlumi,int &nextlumi,int &nthlumi){
  if (nthlumi <= nextlumi) return;

  int currentlumi = nextlumi;
  edm::LogInfo("LS|BX|BeamMonitorBx") << "Lumi of the current fit: " << currentlumi << endl;
  lastlumi = currentlumi;
  endLumiOfBSFit_ = currentlumi;

  edm::LogInfo("BX|BeamMonitorBx") << "Time lapsed = " << tmpTime - refTime << std:: endl;

  if (resetHistos_) {
    edm::LogInfo("BX|BeamMonitorBx") << "Resetting Histograms" << endl;
    theBeamFitter->resetCutFlow();
    resetHistos_ = false;
  }
  
  if (fitNLumi_ > 0)
    if (currentlumi%fitNLumi_!=0) return;

  int * fitLS = theBeamFitter->getFitLSRange();
  edm::LogInfo("LS|BX|BeamMonitorBx") << " [Fitter] Do BeamSpot Fit for LS = " << fitLS[0];
  edm::LogInfo("LS|BX|BeamMonitorBx") << " to " << fitLS[1] << endl;
  edm::LogInfo("LS|BX|BeamMonitorBx") << " [BX] Do BeamSpot Fit for LS = " << beginLumiOfBSFit_;
  edm::LogInfo("LS|BX|BeamMonitorBx") << " to " << endLumiOfBSFit_ << endl;

  if (theBeamFitter->runPVandTrkFitter()) {
    std::map< int, reco::BeamSpot> bsmap = theBeamFitter->getBeamSpotMap();
    edm::LogInfo("BX|BeamMonitorBx") << "Number of bx = " << bsmap.size() << endl;
    if (countBx_ < bsmap.size()) {
      countBx_ = bsmap.size();
      BookHistos(countBx_,varMap);
    }

    int * LSRange = theBeamFitter->getFitLSRange();
    char tmpTitle[50];
    sprintf(tmpTitle,"%s %i %s %i %s"," [cm] (LS: ",LSRange[0]," to ",LSRange[1],")");
    for (std::map<std::string,std::string>::const_iterator varName = varMap.begin();
	 varName != varMap.end(); ++varName) {
      hs[varName->first]->setTitle(varName->second + " " + tmpTitle);
      hs[varName->first]->Reset();
    }

    int nthBin = countBx_;
    for (std::map<int,reco::BeamSpot>::const_iterator abspot = bsmap.begin();
	 abspot!= bsmap.end(); ++abspot,nthBin--) {
      reco::BeamSpot bs = abspot->second;
      int bx = abspot->first;
      for (std::map<std::string,std::string>::const_iterator varName = varMap.begin();
	   varName != varMap.end(); ++varName) {
	hs[varName->first]->setBinContent(1,nthBin,bx);
      }
      hs["x0_bx"]->setBinContent(2,nthBin,bs.x0());
      hs["y0_bx"]->setBinContent(2,nthBin,bs.y0());
      hs["z0_bx"]->setBinContent(2,nthBin,bs.z0());
      hs["sigmaZ_bx"]->setBinContent(2,nthBin,bs.sigmaZ());
      hs["sigmaX_bx"]->setBinContent(2,nthBin,bs.BeamWidthX());
      hs["sigmaY_bx"]->setBinContent(2,nthBin,bs.BeamWidthY());

      hs["x0_bx"]->setBinContent(3,nthBin,bs.x0Error());
      hs["y0_bx"]->setBinContent(3,nthBin,bs.y0Error());
      hs["z0_bx"]->setBinContent(3,nthBin,bs.z0Error());
      hs["sigmaZ_bx"]->setBinContent(3,nthBin,bs.sigmaZ0Error());
      hs["sigmaX_bx"]->setBinContent(3,nthBin,bs.BeamWidthXError());
      hs["sigmaY_bx"]->setBinContent(3,nthBin,bs.BeamWidthYError());
      BookTrendHistos(false,bx,varMap1,"FitBx","Trending","bx");
      FillTrendHistos(bx,varMap1,bs);
    }
  }
  //   else
  //     edm::LogInfo("BeamMonitorBx") << "Bad Fit!!!" << endl;

  if (resetFitNLumi_ > 0 && currentlumi%resetFitNLumi_ == 0) {
    edm::LogInfo("LS|BX|BeamMonitorBx") << "Reset track collection for beam fit!!!" <<endl;
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
void BeamMonitorBx::endRun(const Run& r, const EventSetup& context){

}

//--------------------------------------------------------
void BeamMonitorBx::endJob(const LuminosityBlock& lumiSeg, 
			   const EventSetup& iSetup){
}

DEFINE_FWK_MODULE(BeamMonitorBx);
