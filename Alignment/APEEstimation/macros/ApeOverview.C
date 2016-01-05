// Usage:
// .L ApeOverview.C++
// ApeOverview a1("test.root");
// a1.getOverview()
// a1.printOverview() resp. a1.printOverview("apeOverview.ps",ApeOverview::event)


#include "ApeOverview.h"

#include <sstream>    //for stringstream
#include <iostream>   //for cout

#include "TProfile.h"
#include "TPostScript.h"
#include "TLatex.h"


ApeOverview::ApeOverview(const TString inputFileName):
inputFile_(TFile::Open(inputFileName)), moduleNo_(1), onlyZoomedHists_(false), firstSelectedSector_("")
{
  eventPadCounter_.first = eventPadCounter_.second = trackPadCounter_.first = trackPadCounter_.second = sectorCounter_ = 1;
}

ApeOverview::~ApeOverview(){}



void
ApeOverview::whichModuleInFile(int moduleNo){
  moduleNo_ = moduleNo;
}



void
ApeOverview::onlyZoomedHists(){
  onlyZoomedHists_ = true;
}



void
ApeOverview::setSectorsForOverview(const TString& sectors){
  //std::vector<>
  TObjArray* a_Sector = TString(sectors).Tokenize(",");
  for(Int_t iSec= 0; iSec < a_Sector->GetEntriesFast(); ++iSec){
    
    const TString& sectorNumber = a_Sector->At(iSec)->GetName();
    bool validSectorArgument(false);
    for(unsigned int i = 0; i<20000 ; ++i){
      std::stringstream ssSec;
      ssSec << i;
      if(ssSec.str().c_str() == sectorNumber){
        vSelectedSector_.push_back(i);
	std::cout<<"\n\tPrint overview for Sector:\t"<<sectorNumber<<"\n";
	validSectorArgument = true;
	break;
      }
    }
    if(!validSectorArgument)std::cout<<"\n\tWrong argument in comma separated sector list:\t"<<sectorNumber<<"\n";
  }
}



void
ApeOverview::getOverview(){
  //pluginDir_ = "ApeEstimatorCosmics1/";
  //int nKeys = inputFile_->GetNkeys();
  //pluginDir_ = inputFile_->GetListOfKeys()->At(0)->GetName();
  //pluginDir_ += "/";
  if(moduleNo_<1 || moduleNo_>inputFile_->GetNkeys()){
    std::cout<<"Incorrect number given in method whichModuleInFile(...), cannot continue!!!\n";
    return;
  }
  pluginDir_ = inputFile_->GetListOfKeys()->At(moduleNo_-1)->GetName();
  pluginDir_ += "/";
  
  TDirectory* secDir(0);
  bool sectorBool(true);
  for(unsigned int iSec(1);sectorBool;++iSec){
    std::stringstream sectorName, fullDirectoryName;
    sectorName << "Sector_" << iSec << "/";
    fullDirectoryName << pluginDir_ << sectorName.str();
    TString fullName(fullDirectoryName.str().c_str());
    inputFile_->cd();
    secDir = (TDirectory*)inputFile_->TDirectory::GetDirectory(fullName);
    if(secDir){
      mSectorPadCounter_[iSec].first = mSectorPadCounter_[iSec].second = 1;
      mSectorPair_[iSec].first = mSectorPair_[iSec].second = std::vector<TCanvas*>();
    }
    else sectorBool = false;
  }
  
  // first print event and track histos (FIXME: now contains also sector hists -> rename)
  this->eventAndTrackHistos();
}


TString
ApeOverview::setCanvasName()const{
  std::stringstream canvas;
  int canvasCounter(0);
  canvasCounter += eventPair_.first.size() + eventPair_.second.size() + trackPair_.first.size() + trackPair_.second.size();
  for(std::map<unsigned int,CanvasPair>::const_iterator iSec = mSectorPair_.begin(); iSec != mSectorPair_.end(); ++iSec){
    canvasCounter += (*iSec).second.first.size() + (*iSec).second.second.size();
  }
  canvas << "canvas" << canvasCounter;
  return (canvas.str()).c_str();
}


void
ApeOverview::eventAndTrackHistos(){
  TString histName;
  
  histDir_ = "EventVariables/";
  histLevel_ = event;
  std::cout<<"List of event histograms to print:\n";
  
  this->drawHistToPad("h_trackSize",true);  //logScale only for 1d-hists? Also for Profiles, but not TH2?
  this->drawHistToPad("h_trackSizeGood",true);  //logScale only for 1d-hists? Also for Profiles, but not TH2?
  
  
  histDir_ = "TrackVariables/";
  histLevel_ = track;
  std::cout<<"List of track histograms to print:\n";
  
  if(!onlyZoomedHists_){
  this->drawHistToPad("h_hitsSize");
  this->drawHistToPad("h_hitsValid");
  this->drawHistToPad("h_hitsGood");
  this->drawHistToPad("h_hits2D");
  this->drawHistToPad("h_hitsInvalid");
  this->drawHistToPad("h_layersMissed");
  this->drawHistToPad("h_hitsPixel");
  this->drawHistToPad("h_hitsStrip");
  this->setNewCanvas(dim1);
  }
  
  this->drawHistToPad("h_chi2");
  this->drawHistToPad("h_ndof");
  this->drawHistToPad("h_norChi2");
  if(!onlyZoomedHists_){
  this->drawHistToPad("h_eta");
  this->drawHistToPad("h_theta");
  this->drawHistToPad("h_phi");
  }
  this->setNewCanvas(dim1);
  
  if(!onlyZoomedHists_){
  this->drawHistToPad("h_etaErr");
  this->drawHistToPad("h_phiErr");
  this->drawHistToPad("h_ptErr");
  this->drawHistToPad("h_etaSig");
  this->drawHistToPad("h_phiSig");
  this->drawHistToPad("h_ptSig");
  }
  this->setNewCanvas(dim1);
  
  this->drawHistToPad("h_pt");
  this->drawHistToPad("h_p");
  this->drawHistToPad("h_prob");
  if(!onlyZoomedHists_){
  this->drawHistToPad("h_charge",false);
  this->drawHistToPad("h_meanAngle");
  }
  this->setNewCanvas(dim1);
  
  this->drawHistToPad("h_d0Beamspot");
  if(!onlyZoomedHists_){
  this->drawHistToPad("h_d0BeamspotErr");
  this->drawHistToPad("h_d0BeamspotSig");
  }
  this->drawHistToPad("h_dz");
  if(!onlyZoomedHists_){
  this->drawHistToPad("h_dzErr");
  this->drawHistToPad("h_dzSig");
  }
  this->setNewCanvas(dim1);
  
  this->drawHistToPad("h2_hitsPixelVsEta",false);
  this->drawHistToPad("h2_hitsStripVsEta",false);
  this->drawHistToPad("h2_ptVsEta",false);
  this->drawHistToPad("h2_hitsPixelVsTheta",false);
  this->drawHistToPad("h2_hitsStripVsTheta",false);
  this->drawHistToPad("h2_ptVsTheta",false);
  
  if(!onlyZoomedHists_){
  this->drawHistToPad("h2_hitsGoodVsHitsValid",false);
  this->drawHistToPad("h2_meanAngleVsHits",false);
  this->setNewCanvas(dim2);
  }
  
  histLevel_ = sector;
  TDirectory* secDir(0);
  bool sectorBool(true);
  for(unsigned int iSec(1);sectorBool;++iSec){
    std::stringstream sectorName, fullDirectoryName;
    sectorName << "Sector_" << iSec << "/";
    fullDirectoryName << pluginDir_ << sectorName.str();
    TString fullName(fullDirectoryName.str().c_str());
    inputFile_->cd();
    secDir = (TDirectory*)inputFile_->TDirectory::GetDirectory(fullName);
    sectorCounter_ = iSec;
    if(secDir){
      
      bool selectedSector(false);
      std::stringstream ssFirstSelectedSector;
      ssFirstSelectedSector << "Sector_";
      if(vSelectedSector_.size() == 0){
        selectedSector = true;
	ssFirstSelectedSector << "1";
      }
      else{
        ssFirstSelectedSector << *(vSelectedSector_.begin());
      }
      firstSelectedSector_ = ssFirstSelectedSector.str().c_str();
      
      for(std::vector<unsigned int>::const_iterator iSelSec = vSelectedSector_.begin(); iSelSec != vSelectedSector_.end(); ++iSelSec){
        if(iSec==*iSelSec)selectedSector = true;
      }
      if(!selectedSector)continue;
      
      histDir_  = sectorName.str().c_str();
      if(histDir_.BeginsWith(firstSelectedSector_))std::cout<<"List of hit histograms to print (showcase for "<<firstSelectedSector_<<", all other sectors identical):\n";
      
      this->drawHistToPad("sectorTitleSheet");
      
      
      // Cluster Parameters 1D
      
      this->drawHistToPad("h_WidthX");
      this->drawHistToPad("h_WidthProj");
      this->drawHistToPad("h_WidthDiff");
      if(!onlyZoomedHists_){
      this->drawHistToPad("h_MaxStrip",false);
      this->drawHistToPad("h_MaxIndex");
      this->drawHistToPad("h_BaryStripX",false);
      }
      this->drawHistToPad("h_ChargeStrip");
      this->drawHistToPad("h_ChargePixel");
      this->drawHistToPad("h_SOverN");
      if(!onlyZoomedHists_){
      this->drawHistToPad("h_MaxCharge");
      this->drawHistToPad("h_IsOnEdge");
      this->drawHistToPad("h_HasBadPixels");
      this->drawHistToPad("h_SpansTwoRoc");
      this->setNewCanvas(dim1);
      this->drawHistToPad("h_ChargeOnEdges");
      this->drawHistToPad("h_ChargeAsymmetry");
      this->drawHistToPad("h_ChargeLRminus");
      this->drawHistToPad("h_ChargeLRplus");
      this->drawHistToPad("h_ClusterProbXY");
      this->drawHistToPad("h_ClusterProbQ");
      this->drawHistToPad("h_ClusterProbXYQ");
      }
      this->drawHistToPad("h_LogClusterProb");
      if(!onlyZoomedHists_){
      this->drawHistToPad("h_QBin");
      }
      this->setNewCanvas(dim1);
      
      this->drawHistToPad("h_WidthY_y");
      if(!onlyZoomedHists_){
      this->drawHistToPad("h_BaryStripY_y",false);
      }
      this->drawHistToPad("h_ChargePixel_y");
      if(!onlyZoomedHists_){
      this->drawHistToPad("h_IsOnEdge_y");
      this->drawHistToPad("h_HasBadPixels_y");
      this->drawHistToPad("h_SpansTwoRoc_y");
      this->setNewCanvas(dim1);
      this->drawHistToPad("h_ClusterProbXY_y");
      this->drawHistToPad("h_ClusterProbQ_y");
      this->drawHistToPad("h_ClusterProbXYQ_y");
      }
      this->drawHistToPad("h_LogClusterProb_y");
      if(!onlyZoomedHists_){
      this->drawHistToPad("h_QBin_y");
      }
      this->setNewCanvas(dim1);
      
      
      // Hit parameters 1D
      
      this->drawHistToPad("h_ResX");
      this->drawHistToPad("h_NorResX");
      this->drawHistToPad("h_ProbX",false);
      this->drawHistToPad("h_SigmaXHit");
      this->drawHistToPad("h_SigmaXTrk");
      this->drawHistToPad("h_SigmaX");
      for(unsigned int iUint = 1;;++iUint){
        std::stringstream sigmaXHit, sigmaXTrk, sigmaX;
        sigmaXHit << "h_sigmaXHit_" << iUint;
        sigmaXTrk << "h_sigmaXTrk_" << iUint;
        sigmaX    << "h_sigmaX_"    << iUint;
	if(this->drawHistToPad(sigmaXHit.str().c_str()) == -1){this->setNewCanvas(dim1); break;}
	this->drawHistToPad(sigmaXTrk.str().c_str());
	this->drawHistToPad(sigmaX.str().c_str());
      }
      if(!onlyZoomedHists_){
      this->drawHistToPad("h_XHit");
      this->drawHistToPad("h_XTrk");
      this->setNewCanvas(dim1);
      this->drawHistToPad("h_PhiSens");
      this->drawHistToPad("h_PhiSensX");
      this->drawHistToPad("h_PhiSensY");
      }
      this->setNewCanvas(dim1);
      
      this->drawHistToPad("h_ResY");
      this->drawHistToPad("h_NorResY");
      this->drawHistToPad("h_ProbY",false);
      this->drawHistToPad("h_SigmaYHit_y");
      this->drawHistToPad("h_SigmaYTrk_y");
      this->drawHistToPad("h_SigmaY_y");
      for(unsigned int iUint = 1;;++iUint){
        std::stringstream sigmaXHit, sigmaXTrk, sigmaX;
        sigmaXHit << "h_sigmaYHit_" << iUint;
        sigmaXTrk << "h_sigmaYTrk_" << iUint;
        sigmaX    << "h_sigmaY_"    << iUint;
	if(this->drawHistToPad(sigmaXHit.str().c_str()) == -1){this->setNewCanvas(dim1); break;}
	this->drawHistToPad(sigmaXTrk.str().c_str());
	this->drawHistToPad(sigmaX.str().c_str());
      }
      if(!onlyZoomedHists_){
      this->drawHistToPad("h_YHit");
      this->drawHistToPad("h_YTrk");
      this->setNewCanvas(dim1);
      this->drawHistToPad("h_PhiSens_y");
      this->drawHistToPad("h_PhiSensX_y");
      this->drawHistToPad("h_PhiSensY_y");
      }
      this->setNewCanvas(dim1);
      
      
      // Track, Cluster, and Hit parameters 2D
      
      // vs. hit, track or residual error
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaXTrkVsHitsValid",false);
      this->drawHistToPad("h2_sigmaXTrkVsHitsGood",false);
      this->drawHistToPad("h2_sigmaXTrkVsMeanAngle",false);
      this->setNewCanvas(dim2);
      }
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaXTrkVsHitsInvalid",false);
      this->drawHistToPad("h2_sigmaXTrkVsLayersMissed",false);
      this->drawHistToPad("h2_sigmaXTrkVsHits2D",false);
      this->drawHistToPad("h2_sigmaXTrkVsHitsPixel",false);
      this->drawHistToPad("h2_sigmaXTrkVsHitsStrip",false);
      }
      this->drawHistToPad("h2_sigmaXTrkVsP",false);
      this->setNewCanvas(dim2);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaXTrkVsTheta",false);
      this->drawHistToPad("h2_sigmaXTrkVsPhi",false);
      this->drawHistToPad("h2_sigmaXTrkVsMaxStrip",false);
      this->setNewCanvas(dim2);
      }
      this->drawHistToPad("h2_sigmaXTrkVsD0Beamspot",false);
      this->drawHistToPad("h2_sigmaXTrkVsDz",false);
      this->drawHistToPad("h2_sigmaXTrkVsPt",false);
      this->drawHistToPad("h2_sigmaXTrkVsInvP",false);
      this->drawHistToPad("h2_sigmaXVsNorChi2",false);
      this->setNewCanvas(dim2);
      
      this->drawHistToPad("h2_sigmaXHitVsWidthX",false);
      this->drawHistToPad("h2_sigmaXHitVsWidthProj",false);
      this->drawHistToPad("h2_sigmaXHitVsWidthDiff",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaXHitVsMaxStrip",false);
      this->drawHistToPad("h2_sigmaXHitVsMaxIndex",false);
      this->drawHistToPad("h2_sigmaXHitVsBaryStripX",false);
      }
      this->drawHistToPad("h2_sigmaXHitVsChargeStrip",false);
      this->drawHistToPad("h2_sigmaXHitVsSOverN",false);
      this->drawHistToPad("h2_sigmaXHitVsChargePixel",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaXHitVsMaxCharge",false);
      this->drawHistToPad("h2_sigmaXHitVsChargeOnEdges",false);
      this->drawHistToPad("h2_sigmaXHitVsChargeAsymmetry",false);
      this->drawHistToPad("h2_sigmaXHitVsChargeLRplus",false);
      this->drawHistToPad("h2_sigmaXHitVsChargeLRminus",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_sigmaXHitVsIsOnEdge",false);
      this->drawHistToPad("h2_sigmaXHitVsHasBadPixels",false);
      this->drawHistToPad("h2_sigmaXHitVsSpansTwoRoc",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_sigmaXHitVsClusterProbXY",false);
      this->drawHistToPad("h2_sigmaXHitVsClusterProbQ",false);
      this->drawHistToPad("h2_sigmaXHitVsClusterProbXYQ",false);
      }
      this->drawHistToPad("h2_sigmaXHitVsLogClusterProb",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaXHitVsQBin",false);
      }
      this->setNewCanvas(dim2);
      
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaXHitVsPhiSens",false);
      this->drawHistToPad("h2_sigmaXHitVsPhiSensX",false);
      this->drawHistToPad("h2_sigmaXHitVsPhiSensY",false);
      this->drawHistToPad("h2_sigmaXTrkVsPhiSens",false);
      this->drawHistToPad("h2_sigmaXTrkVsPhiSensX",false);
      this->drawHistToPad("h2_sigmaXTrkVsPhiSensY",false);
      this->drawHistToPad("h2_sigmaXVsPhiSens",false);
      this->drawHistToPad("h2_sigmaXVsPhiSensX",false);
      this->drawHistToPad("h2_sigmaXVsPhiSensY",false);
      }
      
      // vs. normalised residual
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResXVsHitsValid",false);
      this->drawHistToPad("h2_norResXVsHitsGood",false);
      this->drawHistToPad("h2_norResXVsMeanAngle",false);
      this->drawHistToPad("h2_norResXVsHitsInvalid",false);
      this->drawHistToPad("h2_norResXVsLayersMissed",false);
      this->drawHistToPad("h2_norResXVsHits2D",false);
      this->drawHistToPad("h2_norResXVsHitsPixel",false);
      this->drawHistToPad("h2_norResXVsHitsStrip",false);
      }
      this->drawHistToPad("h2_norResXVsP",false);
      if(!onlyZoomedHists_){
      this->setNewCanvas(dim2);
      }
      
      this->drawHistToPad("h2_norResXVsNorChi2",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResXVsTheta",false);
      this->drawHistToPad("h2_norResXVsPhi",false);
      }
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_norResXVsD0Beamspot",false);
      this->drawHistToPad("h2_norResXVsDz",false);
      this->drawHistToPad("h2_norResXVsPt",false);
      
      this->drawHistToPad("h2_norResXVsWidthX",false);
      this->drawHistToPad("h2_norResXVsWidthProj",false);
      this->drawHistToPad("h2_norResXVsWidthDiff",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResXVsMaxStrip",false);
      this->drawHistToPad("h2_norResXVsMaxIndex",false);
      this->drawHistToPad("h2_norResXVsBaryStripX",false);
      }
      this->drawHistToPad("h2_norResXVsChargeStrip",false);
      this->drawHistToPad("h2_norResXVsSOverN",false);
      this->drawHistToPad("h2_norResXVsChargePixel",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResXVsMaxCharge",false);
      this->drawHistToPad("h2_norResXVsChargeOnEdges",false);
      this->drawHistToPad("h2_norResXVsChargeAsymmetry",false);
      this->drawHistToPad("h2_norResXVsChargeLRplus",false);
      this->drawHistToPad("h2_norResXVsChargeLRminus",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_norResXVsIsOnEdge",false);
      this->drawHistToPad("h2_norResXVsHasBadPixels",false);
      this->drawHistToPad("h2_norResXVsSpansTwoRoc",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_norResXVsClusterProbXY",false);
      this->drawHistToPad("h2_norResXVsClusterProbQ",false);
      this->drawHistToPad("h2_norResXVsClusterProbXYQ",false);
      }
      this->drawHistToPad("h2_norResXVsLogClusterProb",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResXVsQBin",false);
      }
      this->setNewCanvas(dim2);
      
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResXVsPhiSens",false);
      this->drawHistToPad("h2_norResXVsPhiSensX",false);
      this->drawHistToPad("h2_norResXVsPhiSensY",false);
      }
      
      this->drawHistToPad("h2_norResXVsSigmaXHit",false);
      this->drawHistToPad("h2_norResXVsSigmaXTrk",false);
      this->drawHistToPad("h2_norResXVsSigmaX",false);
      this->setNewCanvas(dim2);
      
      // vs. probability
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probXVsHitsValid",false);
      this->drawHistToPad("h2_probXVsHitsGood",false);
      this->drawHistToPad("h2_probXVsMeanAngle",false);
      this->drawHistToPad("h2_probXVsHitsInvalid",false);
      this->drawHistToPad("h2_probXVsLayersMissed",false);
      this->drawHistToPad("h2_probXVsHits2D",false);
      this->drawHistToPad("h2_probXVsHitsPixel",false);
      this->drawHistToPad("h2_probXVsHitsStrip",false);
      }
      this->drawHistToPad("h2_probXVsP",false);
      if(!onlyZoomedHists_){
      this->setNewCanvas(dim2);
      }
      
      this->drawHistToPad("h2_probXVsNorChi2",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probXVsTheta",false);
      this->drawHistToPad("h2_probXVsPhi",false);
      }
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_probXVsD0Beamspot",false);
      this->drawHistToPad("h2_probXVsDz",false);
      this->drawHistToPad("h2_probXVsPt",false);
      
      this->drawHistToPad("h2_probXVsWidthX",false);
      this->drawHistToPad("h2_probXVsWidthProj",false);
      this->drawHistToPad("h2_probXVsWidthDiff",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probXVsMaxStrip",false);
      this->drawHistToPad("h2_probXVsMaxIndex",false);
      this->drawHistToPad("h2_probXVsBaryStripX",false);
      }
      this->drawHistToPad("h2_probXVsChargeStrip",false);
      this->drawHistToPad("h2_probXVsSOverN",false);
      this->drawHistToPad("h2_probXVsChargePixel",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probXVsMaxCharge",false);
      this->drawHistToPad("h2_probXVsChargeOnEdges",false);
      this->drawHistToPad("h2_probXVsChargeAsymmetry",false);
      this->drawHistToPad("h2_probXVsChargeLRplus",false);
      this->drawHistToPad("h2_probXVsChargeLRminus",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_probXVsIsOnEdge",false);
      this->drawHistToPad("h2_probXVsHasBadPixels",false);
      this->drawHistToPad("h2_probXVsSpansTwoRoc",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_probXVsClusterProbXY",false);
      this->drawHistToPad("h2_probXVsClusterProbQ",false);
      this->drawHistToPad("h2_probXVsClusterProbXYQ",false);
      }
      this->drawHistToPad("h2_probXVsLogClusterProb",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probXVsQBin",false);
      }
      this->setNewCanvas(dim2);
      
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probXVsPhiSens",false);
      this->drawHistToPad("h2_probXVsPhiSensX",false);
      this->drawHistToPad("h2_probXVsPhiSensY",false);
      }
      
      this->drawHistToPad("h2_probXVsSigmaXHit",false);
      this->drawHistToPad("h2_probXVsSigmaXTrk",false);
      this->drawHistToPad("h2_probXVsSigmaX",false);
      this->setNewCanvas(dim2);
      
      // other
      this->drawHistToPad("h2_widthVsPhiSensX",false);
      this->drawHistToPad("h2_widthVsWidthProj",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_widthDiffVsMaxStrip",false);
      this->drawHistToPad("h2_widthDiffVsSigmaXHit",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_phiSensXVsBarycentreX",false);
      }
      this->setNewCanvas(dim2);
      
      
      // vs. hit, track or residual error (y coordinate)
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaYTrkVsHitsValid",false);
      this->drawHistToPad("h2_sigmaYTrkVsHitsGood",false);
      this->drawHistToPad("h2_sigmaYTrkVsMeanAngle",false);
      this->setNewCanvas(dim2);
      }
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaYTrkVsHitsInvalid",false);
      this->drawHistToPad("h2_sigmaYTrkVsLayersMissed",false);
      this->drawHistToPad("h2_sigmaYTrkVsHits2D",false);
      this->drawHistToPad("h2_sigmaYTrkVsHitsPixel",false);
      this->drawHistToPad("h2_sigmaYTrkVsHitsStrip",false);
      }
      this->drawHistToPad("h2_sigmaYTrkVsP",false);
      this->setNewCanvas(dim2);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaYTrkVsTheta",false);
      this->drawHistToPad("h2_sigmaYTrkVsPhi",false);
      this->drawHistToPad("h2_sigmaYTrkVsMaxStrip",false);
      this->setNewCanvas(dim2);
      }
      this->drawHistToPad("h2_sigmaYTrkVsD0Beamspot",false);
      this->drawHistToPad("h2_sigmaYTrkVsDz",false);
      this->drawHistToPad("h2_sigmaYTrkVsPt",false);
      this->drawHistToPad("h2_sigmaYTrkVsInvP",false);
      this->drawHistToPad("h2_sigmaYVsNorChi2",false);
      this->setNewCanvas(dim2);
      
      this->drawHistToPad("h2_sigmaYHitVsWidthY",false);
      this->drawHistToPad("h2_sigmaYHitVsWidthProj",false);
      this->drawHistToPad("h2_sigmaYHitVsWidthDiff",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaYHitVsMaxStrip",false);
      this->drawHistToPad("h2_sigmaYHitVsMaxIndex",false);
      this->drawHistToPad("h2_sigmaYHitVsBaryStripY",false);
      }
      this->drawHistToPad("h2_sigmaYHitVsChargeStrip",false);
      this->drawHistToPad("h2_sigmaYHitVsSOverN",false);
      this->drawHistToPad("h2_sigmaYHitVsChargePixel",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaYHitVsMaxCharge",false);
      this->drawHistToPad("h2_sigmaYHitVsChargeOnEdges",false);
      this->drawHistToPad("h2_sigmaYHitVsChargeAsymmetry",false);
      this->drawHistToPad("h2_sigmaYHitVsChargeLRplus",false);
      this->drawHistToPad("h2_sigmaYHitVsChargeLRminus",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_sigmaYHitVsIsOnEdge",false);
      this->drawHistToPad("h2_sigmaYHitVsHasBadPixels",false);
      this->drawHistToPad("h2_sigmaYHitVsSpansTwoRoc",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_sigmaYHitVsClusterProbXY",false);
      this->drawHistToPad("h2_sigmaYHitVsClusterProbQ",false);
      this->drawHistToPad("h2_sigmaYHitVsClusterProbXYQ",false);
      }
      this->drawHistToPad("h2_sigmaYHitVsLogClusterProb",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaYHitVsQBin",false);
      }
      this->setNewCanvas(dim2);
      
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_sigmaYHitVsPhiSens",false);
      this->drawHistToPad("h2_sigmaYHitVsPhiSensX",false);
      this->drawHistToPad("h2_sigmaYHitVsPhiSensY",false);
      this->drawHistToPad("h2_sigmaYTrkVsPhiSens",false);
      this->drawHistToPad("h2_sigmaYTrkVsPhiSensX",false);
      this->drawHistToPad("h2_sigmaYTrkVsPhiSensY",false);
      this->drawHistToPad("h2_sigmaYVsPhiSens",false);
      this->drawHistToPad("h2_sigmaYVsPhiSensX",false);
      this->drawHistToPad("h2_sigmaYVsPhiSensY",false);
      }
      
      // vs. normalised residual (y coordinate)
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResYVsHitsValid",false);
      this->drawHistToPad("h2_norResYVsHitsGood",false);
      this->drawHistToPad("h2_norResYVsMeanAngle",false);
      this->drawHistToPad("h2_norResYVsHitsInvalid",false);
      this->drawHistToPad("h2_norResYVsLayersMissed",false);
      this->drawHistToPad("h2_norResYVsHits2D",false);
      this->drawHistToPad("h2_norResYVsHitsPixel",false);
      this->drawHistToPad("h2_norResYVsHitsStrip",false);
      }
      this->drawHistToPad("h2_norResYVsP",false);
      if(!onlyZoomedHists_){
      this->setNewCanvas(dim2);
      }
      
      this->drawHistToPad("h2_norResYVsNorChi2",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResYVsTheta",false);
      this->drawHistToPad("h2_norResYVsPhi",false);
      }
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_norResYVsD0Beamspot",false);
      this->drawHistToPad("h2_norResYVsDz",false);
      this->drawHistToPad("h2_norResYVsPt",false);
      
      this->drawHistToPad("h2_norResYVsWidthY",false);
      this->drawHistToPad("h2_norResYVsWidthProj",false);
      this->drawHistToPad("h2_norResYVsWidthDiff",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResYVsMaxStrip",false);
      this->drawHistToPad("h2_norResYVsMaxIndex",false);
      this->drawHistToPad("h2_norResYVsBaryStripY",false);
      }
      this->drawHistToPad("h2_norResYVsChargeStrip",false);
      this->drawHistToPad("h2_norResYVsSOverN",false);
      this->drawHistToPad("h2_norResYVsChargePixel",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResYVsMaxCharge",false);
      this->drawHistToPad("h2_norResYVsChargeOnEdges",false);
      this->drawHistToPad("h2_norResYVsChargeAsymmetry",false);
      this->drawHistToPad("h2_norResYVsChargeLRplus",false);
      this->drawHistToPad("h2_norResYVsChargeLRminus",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_norResYVsIsOnEdge",false);
      this->drawHistToPad("h2_norResYVsHasBadPixels",false);
      this->drawHistToPad("h2_norResYVsSpansTwoRoc",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_norResYVsClusterProbXY",false);
      this->drawHistToPad("h2_norResYVsClusterProbQ",false);
      this->drawHistToPad("h2_norResYVsClusterProbXYQ",false);
      }
      this->drawHistToPad("h2_norResYVsLogClusterProb",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResYVsQBin",false);
      }
      this->setNewCanvas(dim2);
      
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_norResYVsPhiSens",false);
      this->drawHistToPad("h2_norResYVsPhiSensX",false);
      this->drawHistToPad("h2_norResYVsPhiSensY",false);
      }
      
      this->drawHistToPad("h2_norResYVsSigmaYHit",false);
      this->drawHistToPad("h2_norResYVsSigmaYTrk",false);
      this->drawHistToPad("h2_norResYVsSigmaY",false);
      this->setNewCanvas(dim2);
      
      // vs. probability (y coordinate)
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probYVsHitsValid",false);
      this->drawHistToPad("h2_probYVsHitsGood",false);
      this->drawHistToPad("h2_probYVsMeanAngle",false);
      this->drawHistToPad("h2_probYVsHitsInvalid",false);
      this->drawHistToPad("h2_probYVsLayersMissed",false);
      this->drawHistToPad("h2_probYVsHits2D",false);
      this->drawHistToPad("h2_probYVsHitsPixel",false);
      this->drawHistToPad("h2_probYVsHitsStrip",false);
      }
      this->drawHistToPad("h2_probYVsP",false);
      if(!onlyZoomedHists_){
      this->setNewCanvas(dim2);
      }
      
      this->drawHistToPad("h2_probYVsNorChi2",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probYVsTheta",false);
      this->drawHistToPad("h2_probYVsPhi",false);
      }
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_probYVsD0Beamspot",false);
      this->drawHistToPad("h2_probYVsDz",false);
      this->drawHistToPad("h2_probYVsPt",false);
      
      this->drawHistToPad("h2_probYVsWidthY",false);
      this->drawHistToPad("h2_probYVsWidthProj",false);
      this->drawHistToPad("h2_probYVsWidthDiff",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probYVsMaxStrip",false);
      this->drawHistToPad("h2_probYVsMaxIndex",false);
      this->drawHistToPad("h2_probYVsBaryStripY",false);
      }
      this->drawHistToPad("h2_probYVsChargeStrip",false);
      this->drawHistToPad("h2_probYVsSOverN",false);
      this->drawHistToPad("h2_probYVsChargePixel",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probYVsMaxCharge",false);
      this->drawHistToPad("h2_probYVsChargeOnEdges",false);
      this->drawHistToPad("h2_probYVsChargeAsymmetry",false);
      this->drawHistToPad("h2_probYVsChargeLRplus",false);
      this->drawHistToPad("h2_probYVsChargeLRminus",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_probYVsIsOnEdge",false);
      this->drawHistToPad("h2_probYVsHasBadPixels",false);
      this->drawHistToPad("h2_probYVsSpansTwoRoc",false);
      this->setNewCanvas(dim2);
      this->drawHistToPad("h2_probYVsClusterProbXY",false);
      this->drawHistToPad("h2_probYVsClusterProbQ",false);
      this->drawHistToPad("h2_probYVsClusterProbXYQ",false);
      }
      this->drawHistToPad("h2_probYVsLogClusterProb",false);
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probYVsQBin",false);
      }
      this->setNewCanvas(dim2);
      
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_probYVsPhiSens",false);
      this->drawHistToPad("h2_probYVsPhiSensX",false);
      this->drawHistToPad("h2_probYVsPhiSensY",false);
      }
      
      this->drawHistToPad("h2_probYVsSigmaYHit",false);
      this->drawHistToPad("h2_probYVsSigmaYTrk",false);
      this->drawHistToPad("h2_probYVsSigmaY",false);
      this->setNewCanvas(dim2);
      
      // other (y coordinate)
      if(!onlyZoomedHists_){
      this->drawHistToPad("h2_phiSensYVsBarycentreY",false);
      }
      this->setNewCanvas(dim2);
      
      
      // Additional 1D histograms
      
      this->setNewCanvas(dim1);
      for(unsigned int iBin(1); ; ++iBin){
        TDirectory* intDir(0);
	std::stringstream intervalName, fullDirectoryName2;
        intervalName << "Interval_" << iBin << "/";
	fullDirectoryName2 << fullDirectoryName.str() << intervalName.str();
        TString fullName2(fullDirectoryName2.str().c_str());
	inputFile_->cd();
        intDir = (TDirectory*)inputFile_->TDirectory::GetDirectory(fullName2);
	if(intDir){
	  histDir_  = (sectorName.str() + intervalName.str()).c_str();
	  if(!onlyZoomedHists_){
	  this->drawHistToPad("h_norResX",false);
	  }
	}
	else break;
      }
      
      this->setNewCanvas(dim1);
      for(unsigned int iBin(1); ; ++iBin){
        TDirectory* intDir(0);
	std::stringstream intervalName, fullDirectoryName2;
        intervalName << "Interval_" << iBin << "/";
	fullDirectoryName2 << fullDirectoryName.str() << intervalName.str();
        TString fullName2(fullDirectoryName2.str().c_str());
	inputFile_->cd();
        intDir = (TDirectory*)inputFile_->TDirectory::GetDirectory(fullName2);
	if(intDir){
	  histDir_  = (sectorName.str() + intervalName.str()).c_str();
	  if(!onlyZoomedHists_){
	  this->drawHistToPad("h_norResY",false);
	  }
	}
	else break;
      }
      
      if(!onlyZoomedHists_){
      this->setNewCanvas(dim1);
      TDirectory* resDir(0);
      std::string resultName("Results/");
      histDir_  = (sectorName.str() + resultName).c_str();
      std::stringstream fullDirectoryName3;
      fullDirectoryName3 << fullDirectoryName.str() << resultName;
      resDir = (TDirectory*)inputFile_->TDirectory::GetDirectory(fullDirectoryName3.str().c_str());
      if(resDir)
      this->drawHistToPad("h_entriesX");
      resultName = "";
      histDir_  = (sectorName.str() + resultName).c_str();
      this->drawHistToPad("h_meanX",false);
      this->drawHistToPad("h_fitMeanX1",false);
      this->drawHistToPad("h_fitMeanX2",false);
      this->drawHistToPad("h_rmsX",false);
      this->drawHistToPad("h_residualWidthX1",false);
      this->drawHistToPad("h_residualWidthX2",false);
      this->drawHistToPad("h_weightX",false);
      this->drawHistToPad("h_correctionX1",false);
      this->drawHistToPad("h_correctionX2",false);
      }
      
      if(!onlyZoomedHists_){
      this->setNewCanvas(dim1);
      TDirectory* resDir(0);
      std::string resultName("Results/");
      histDir_  = (sectorName.str() + resultName).c_str();
      std::stringstream fullDirectoryName3;
      fullDirectoryName3 << fullDirectoryName.str() << resultName;
      resDir = (TDirectory*)inputFile_->TDirectory::GetDirectory(fullDirectoryName3.str().c_str());
      if(resDir)
      this->drawHistToPad("h_entriesY");
      resultName = "";
      histDir_  = (sectorName.str() + resultName).c_str();
      this->drawHistToPad("h_meanY",false);
      this->drawHistToPad("h_fitMeanY1",false);
      this->drawHistToPad("h_fitMeanY2",false);
      this->drawHistToPad("h_rmsY",false);
      this->drawHistToPad("h_residualWidthY1",false);
      this->drawHistToPad("h_residualWidthY2",false);
      this->drawHistToPad("h_weightY",false);
      this->drawHistToPad("h_correctionY1",false);
      this->drawHistToPad("h_correctionY2",false);
      }
    }
    else sectorBool = false;
  }
}



int
ApeOverview::drawHistToPad(const TString histName, const bool setLogScale){
  PadCounterPair* padCounter;
  CanvasPair* canvasPair;
  
  if(histLevel_==event){padCounter=&eventPadCounter_; canvasPair=&eventPair_;}
  else if(histLevel_==track){padCounter=&trackPadCounter_; canvasPair=&trackPair_;}
  else if(histLevel_==sector){padCounter=&mSectorPadCounter_[sectorCounter_]; canvasPair=&mSectorPair_[sectorCounter_];}
  else return -1;
  
  //if(histName.BeginsWith("h_", TString::kIgnoreCase)) not case sensitive
  if(histName.BeginsWith("h_")){
    TH1 *hist1(0);
    inputFile_->GetObject(pluginDir_ + histDir_ + histName + ";1", hist1);
    if(histDir_.BeginsWith(firstSelectedSector_) || histDir_ == "TrackVariables/" || histDir_ == "EventVariables/")std::cout<<"\tDraw 1D Histo\t\t"<<pluginDir_<<histDir_<<histName<<"\n";
    //GetEntries delivers double
    if(hist1){
      if(padCounter->first >= 7){padCounter->first = 1;}
      if(padCounter->first == 1){
        //new TCanvas
        TCanvas* canvas0 = new TCanvas(this->setCanvasName());
        canvas0->Divide(3,2);
        canvasPair->first.push_back(canvas0);
      }
      (*(--(canvasPair->first.end())))->cd(padCounter->first);
      
      if(setLogScale==true && hist1->GetEffectiveEntries()>0.1){
      //if(setLogScale==true && hist1->Integral()>0.1)(*(--(canvasPair->first.end())))->cd(padCounter->first)->SetLogy();  // gives same result
        (*(--(canvasPair->first.end())))->cd(padCounter->first)->SetLogy();
	hist1->SetMinimum(0.5);
      }
      hist1->Draw();
      
      ++(padCounter->first);
    }
    else{if(histDir_.BeginsWith(firstSelectedSector_) || histDir_ == "TrackVariables/" || histDir_ == "EventVariables/")std::cout<<"\t\t\t\t... Histogram does not exist!\n"; return -1;}
    return 0;
  }
  
  else if(histName.BeginsWith("h2_")){
    TH2 *hist2(0);
    inputFile_->GetObject(pluginDir_ + histDir_ + histName + ";1", hist2);
    if(histDir_.BeginsWith(firstSelectedSector_) || histDir_ == "TrackVariables/" || histDir_ == "EventVariables/")std::cout<<"\tDraw 2D Histo\t\t"<<pluginDir_<<histDir_<<histName<<"\n";
    if(hist2){
      if(padCounter->second >= 4){padCounter->second = 1;}
      if(padCounter->second == 1){
        //new TCanvas
        TCanvas* canvas0 = new TCanvas(this->setCanvasName());
        canvas0->Divide(3,2);
        canvasPair->second.push_back(canvas0);
      }
      (*(--(canvasPair->second.end())))->cd(padCounter->second);
      
      if(setLogScale==true && hist2->GetEffectiveEntries()>0.1){
        (*(--(canvasPair->second.end())))->cd(padCounter->second)->SetLogy();
      }
      hist2->Draw("box");
      
      
      // Include profile corresponding to 2D histo automatically here
      TString histNameP(histName);
      histNameP.ReplaceAll("h2_","p_");
      TProfile *histP(0);
      inputFile_->GetObject(pluginDir_ + histDir_ + histNameP + ";1", histP);
      if(histDir_.BeginsWith(firstSelectedSector_) || histDir_ == "TrackVariables/" || histDir_ == "EventVariables/")std::cout<<"\tDraw Profile Histo\t"<<pluginDir_<<histDir_<<histNameP<<"\n";
      if(histP){
        (*(--(canvasPair->second.end())))->cd(padCounter->second+3);
        if(setLogScale==true && histP->GetEffectiveEntries()>0.1){
	  (*(--(canvasPair->second.end())))->cd(padCounter->second+3)->SetLogy();
	}
	
	// Loop for separating mean and RMS per bin (separate entries in final plot)
	TProfile *rmsPlot(0);
	if(histNameP.BeginsWith("p_norResXVs") || histNameP.BeginsWith("p_probXVs") ||
	   histNameP.BeginsWith("p_norResYVs") || histNameP.BeginsWith("p_probYVs")){
	  std::stringstream tempName;
	  tempName << "temp_" << histNameP << "_" << sectorCounter_ << "_" << moduleNo_;
	  TString tempHist(tempName.str().c_str());
	  const int nBinX(histP->GetNbinsX());
	  rmsPlot = new TProfile(tempHist,"temp",nBinX,histP->GetBinLowEdge(1),histP->GetBinLowEdge(nBinX+1));
	  for(int iBin = 0; iBin < nBinX+1; ++iBin){
	    rmsPlot->SetBinContent(iBin, 1000000*histP->GetBinError(iBin));   // Scale by factor 1000000, and same for next line --> hack to hide error bars (hard to get rid of in TProfile)
	    rmsPlot->SetBinEntries(iBin, (histP->GetBinEntries(iBin)<0.1 ? 0 : 1000000));
	    //rmsPlot->SetBinError(iBin, 0.00001);   // Does not do anything !?
	  }
	  //std::cout<<"\t\tBins "<<tempHist<<" "<<nBinX<<" "<<histP->GetBinLowEdge(1)<<" "<<histP->GetBinLowEdge(nBinX+1)<<"\n";
	}
	
        histP->Draw();
        if(rmsPlot){
	  double yMin(histNameP.BeginsWith("p_probXVs") || histNameP.BeginsWith("p_probYVs") ? -0.1 : -3);
	  double yMax(histNameP.BeginsWith("p_probXVs") || histNameP.BeginsWith("p_probYVs") ? 1.1 : 3);
	  histP->SetErrorOption("");
	  histP->GetYaxis()->SetRangeUser(yMin,yMax);
	  const int nBinX(histP->GetNbinsX());
	  for(int iBin = 0; iBin < nBinX+1; ++iBin){
	    if(histP->GetBinContent(iBin)>yMax)histP->SetBinContent(iBin,histP->GetBinEntries(iBin)*yMax);
	    if(histP->GetBinContent(iBin)<yMin)histP->SetBinContent(iBin,histP->GetBinEntries(iBin)*yMin);
	    if(rmsPlot->GetBinContent(iBin)>yMax)rmsPlot->SetBinContent(iBin,rmsPlot->GetBinEntries(iBin)*yMax);
	    if(rmsPlot->GetBinContent(iBin)<yMin)rmsPlot->SetBinContent(iBin,rmsPlot->GetBinEntries(iBin)*yMin);
	  }
	  rmsPlot->SetMarkerColor(2);
	  rmsPlot->SetLineColor(2);
	  //rmsPlot->SetMarkerStyle(24);
	  rmsPlot->Draw("same");
	  //rmsPlot->Draw("hist p same");
	}
	
      }
      else{if(histDir_.BeginsWith(firstSelectedSector_) || histDir_ == "TrackVariables/" || histDir_ == "EventVariables/")std::cout<<"\t\t\t\t... Histogram does not exist!\n";return -1;}
      
      ++(padCounter->second);
    }
    else{if(histDir_.BeginsWith(firstSelectedSector_) || histDir_ == "TrackVariables/" || histDir_ == "EventVariables/")std::cout<<"\t\t\t\t... Histogram does not exist!\n"; return -1;}
    return 0;
  }
  
  else if(histName.BeginsWith("p_")){std::cout<<"\n\tProfile Plot chosen, but set up automatically"<<std::endl; return 1;}
  else if(histName == "sectorTitleSheet"){
    const TString nameHistName("z_name");
    TH1* nameHist(0);
    inputFile_->GetObject(pluginDir_ + histDir_ + nameHistName + ";1", nameHist);
    TString sectorName(histDir_);
    if(nameHist){
      sectorName += "  --  ";
      sectorName += nameHist->GetTitle();
    }
    TCanvas* canvas0 = new TCanvas(this->setCanvasName());
    canvasPair->first.push_back(canvas0);
    (*(--(canvasPair->first.end())))->cd();
    TLatex *title1 = new TLatex(0.1583,0.925,sectorName);title1->SetNDC();//title1->SetTextSize(0.075);
    title1->Draw();
    this->setNewCanvas(dim1);
    return 0;
  }
  else{std::cout<<"\n\tIncorrect Initial Letters for histogram !!!"<<std::endl; return -1;}
}


int
ApeOverview::setNewCanvas(const PlotDimension& pDim){
  std::pair<unsigned int, unsigned int>* padCounter;
  if(histLevel_==event){padCounter=&eventPadCounter_;}
  else if(histLevel_==track){padCounter=&trackPadCounter_;}
  else if(histLevel_==sector){padCounter=&mSectorPadCounter_[sectorCounter_];}
  else return -1;
  
  if(pDim==dim1){padCounter->first = 7; return 0;}
  else if(pDim==dim2){padCounter->second = 4; return 0;}
  else return -1;
  return 0;
}



void
ApeOverview::printOverview(const TString& outputFileName, const HistLevel& histLevel){
  if(eventPair_.first.size()==0 && eventPair_.second.size()==0 &&
     trackPair_.first.size()==0 && trackPair_.second.size()==0 &&
     mSectorPair_.size()==0)return;
  std::cout<<"\tCreate PostScript File:\t"<<outputFileName<<std::endl;
  TPostScript* ps = new TPostScript(outputFileName,112);
  std::vector<TCanvas*>::const_iterator iCan;
  if(histLevel==event){
    for(iCan = eventPair_.first.begin(); iCan != eventPair_.first.end(); ++iCan){ps->NewPage();(*iCan)->Draw();}
    for(iCan = eventPair_.second.begin(); iCan != eventPair_.second.end(); ++iCan){ps->NewPage();(*iCan)->Draw();}
  }
  if(histLevel==event || histLevel==track){
    for(iCan = trackPair_.first.begin(); iCan != trackPair_.first.end(); ++iCan){ps->NewPage();(*iCan)->Draw();}
    for(iCan = trackPair_.second.begin(); iCan != trackPair_.second.end(); ++iCan){ps->NewPage();(*iCan)->Draw();}
  }
  if(histLevel==event || histLevel==track || histLevel==sector){
    std::map<unsigned int, CanvasPair>::const_iterator iSec;
    for(iSec = mSectorPair_.begin(); iSec != mSectorPair_.end(); ++iSec){
      for(iCan = iSec->second.first.begin(); iCan != iSec->second.first.end(); ++iCan){ps->NewPage();(*iCan)->Draw();}
      for(iCan = iSec->second.second.begin(); iCan != iSec->second.second.end(); ++iCan){ps->NewPage();(*iCan)->Draw();}
    }
  }
  ps->Close();
  
  // Now close the canvases printed to the PostScript ...
  for(iCan = eventPair_.first.begin(); iCan != eventPair_.first.end(); ++iCan){(*iCan)->Close();}
  for(iCan = eventPair_.second.begin(); iCan != eventPair_.second.end(); ++iCan){(*iCan)->Close();}
  for(iCan = trackPair_.first.begin(); iCan != trackPair_.first.end(); ++iCan){(*iCan)->Close();}
  for(iCan = trackPair_.second.begin(); iCan != trackPair_.second.end(); ++iCan){(*iCan)->Close();}
  for(std::map<unsigned int, CanvasPair>::const_iterator iSec = mSectorPair_.begin(); iSec != mSectorPair_.end(); ++iSec){
    for(iCan = iSec->second.first.begin(); iCan != iSec->second.first.end(); ++iCan){(*iCan)->Close();}
    for(iCan = iSec->second.second.begin(); iCan != iSec->second.second.end(); ++iCan){(*iCan)->Close();}
  }
  
  // ... and delete the corresponding vectors (which now contain null pointers only) ...
  eventPair_.first.clear();
  eventPair_.second.clear();
  trackPair_.first.clear();
  trackPair_.second.clear();
  mSectorPair_.clear();
  
  // ... and reset the counters
  eventPadCounter_.first = eventPadCounter_.second = trackPadCounter_.first = trackPadCounter_.second = sectorCounter_ = 1;
  mSectorPadCounter_.clear();
}





