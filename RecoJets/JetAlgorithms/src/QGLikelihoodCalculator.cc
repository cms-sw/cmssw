#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"
#include "RecoJets/JetAlgorithms/interface/Bins.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

#include "TMath.h"

#include <map>
using namespace std;


QGLikelihoodCalculator::QGLikelihoodCalculator( TString dataDir, bool chs){

  if( !(dataDir.EndsWith("/")) ) dataDir += "/";

  TString histoFileName = "ReducedHisto_2012.root";
  if(chs) histoFileName = "ReducedHisto_2012_CHS.root";
  histoFile_ = TFile::Open(TString(edm::FileInPath(dataDir + histoFileName).fullPath()));

  nPtBins_ = 21;
  nRhoBins_ = 45 ;
}


float QGLikelihoodCalculator::computeQGLikelihood2012( float pt, float eta, float rho, int nPFCandidates_QC_ptCut, float ptD_QC, float axis2_QC ) {

  // in forward use inclusive 127-4000 bin
  if( fabs(eta)>2.5 && pt>127. ) pt = 128.;

  std::vector<std::string> varName;
  //varName.push_back("nPFCand_QCJet0");
  if( fabs(eta)<2.5 ) {
    varName.push_back("nPFCand_QC_ptCutJet0");
    varName.push_back("ptD_QCJet0");
    varName.push_back("axis2_QCJet0");
  } else {
    varName.push_back("nPFCand_QC_ptCutJet0_F");
    varName.push_back("ptD_QCJet0_F");
    varName.push_back("axis2_QCJet0_F");
  }

  std::vector<float> vars;
  vars.push_back(nPFCandidates_QC_ptCut);
  vars.push_back(ptD_QC);
  vars.push_back(-log(axis2_QC)); //-log

  double RhoBins[100];
  double PtBins[100];

  Bins::getBins_int(Bins::nPtBins+1,PtBins,Bins::Pt0,Bins::Pt1,true);
  PtBins[Bins::nPtBins+1]=Bins::PtLastExtend;
  Bins::getBins_int(Bins::nRhoBins+1,RhoBins,Bins::Rho0,Bins::Rho1,false);

  double ptMin=0.;
  double ptMax=0;
  double rhoMin=0.;
  double rhoMax=0;

  if(Bins::getBin(Bins::nPtBins,PtBins,pt,&ptMin,&ptMax) <0 ) return -1;
  if(Bins::getBin(Bins::nRhoBins,RhoBins,rho,&rhoMin,&rhoMax) <0 ) return -1;

  float Q=1;
  float G=1;
  char histoName[1023];
  ptMin=ceil(ptMin);
  ptMax=ceil(ptMax);
  rhoMin=floor(rhoMin);
  rhoMax=floor(rhoMax);

  for(unsigned int i=0;i<vars.size();i++){
    //get Histo
    float Qi(1),Gi(1),mQ,mG;
    sprintf( histoName, "rhoBins_pt%.0f_%.0f/%s_quark_pt%.0f_%.0f_rho%.0f", ptMin, ptMax,varName[i].c_str(), ptMin, ptMax, rhoMin);
    if( plots_[histoName] == nullptr ) { //first time 
      plots_[histoName]=(TH1F*)histoFile_->Get(histoName); 
      if( plots_[histoName]->GetEntries()<50 ) plots_[histoName]->Rebin(5); // try to make it more stable
      else if( plots_[histoName]->GetEntries()<500 ) plots_[histoName]->Rebin(2); // try to make it more stable
    }
    if( plots_[histoName] == nullptr ) fprintf(stderr,"Histo %s does not exists\n",histoName); //DEBUG
    plots_[ histoName]->Scale(1./plots_[histoName]->Integral("width")); 

    Qi=plots_[histoName]->GetBinContent(plots_[histoName]->FindBin(vars[i]));
    mQ=plots_[histoName]->GetMean();
	
    sprintf( histoName, "rhoBins_pt%.0f_%.0f/%s_gluon_pt%.0f_%.0f_rho%.0f", ptMin, ptMax,varName[i].c_str(), ptMin, ptMax, rhoMin);
    if( plots_[histoName] == nullptr ) { // first time
      plots_[histoName]=(TH1F*)histoFile_->Get(histoName);
      if( plots_[histoName]->GetEntries()<50 ) plots_[histoName]->Rebin(5); // try to make it more stable
      else if( plots_[histoName]->GetEntries()<500 ) plots_[histoName]->Rebin(2); // try to make it more stable
    }
    if( plots_[histoName] == NULL ) fprintf(stderr,"Histo %s does not exists\n",histoName); //DEBUG
    plots_[ histoName]->Scale(1./plots_[histoName]->Integral("width")); 

    Gi=plots_[histoName]->GetBinContent(plots_[histoName]->FindBin(vars[i]));
    mG=plots_[histoName]->GetMean();
	
    float epsilon=0;
    float delta=0.000001;
    if( Qi<=epsilon && Gi<=epsilon){
      if(mQ>mG){
	if(vars[i]>mQ){ Qi=1-delta;Gi=delta;}
	else if(vars[i]<mG){Qi=delta;Gi=1-delta;}
      }
      else if(mQ<mG){
	if(vars[i]<mQ) {Qi=1-delta;Gi=delta;}
	else if(vars[i]>mG){Qi=delta;Gi=1-delta;}
      }
    } 

    Q*=Qi;
    G*=Gi;	
  }

  if(Q==0) return 0;
  return Q/(Q+G);
}
