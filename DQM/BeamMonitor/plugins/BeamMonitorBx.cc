/*
 * \file BeamMonitorBx.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 * $Date: 2010/06/04 07:10:08 $
 * $Revision: 1.2 $
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

  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"FitBx");

  // Results of good fit:
  BookHistos(1,varMap);
}

//--------------------------------------------------------
void BeamMonitorBx::beginRun(const edm::Run& r, const EventSetup& context) {

  ftimestamp = r.beginTime().value();
  tmpTime = ftimestamp >> 32;
  refTime =  tmpTime;
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
  dbe_->setCurrentFolder(monitorName_+"FitBx");

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
