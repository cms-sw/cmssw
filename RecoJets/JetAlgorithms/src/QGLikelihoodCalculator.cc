#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"
#include "RecoJets/JetAlgorithms/interface/Bins.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

#include "TMath.h"

#include <map>
using namespace std;

// constructor:

QGLikelihoodCalculator::QGLikelihoodCalculator( TString dataDir, Bool_t chs){

  if( !(dataDir.EndsWith("/")) ) dataDir += "/";

  TString histoFileName = "ReducedHisto_2012.root";
  if(chs) histoFileName = "ReducedHisto_2012_CHS.root";
//std::cout <<" blabla: " << TString(edm::FileInPath(dataDir + histoFileName).fullPath()) << std::endl;
  histoFile_ = TFile::Open(TString(edm::FileInPath(dataDir + histoFileName).fullPath()));

  nPtBins_ = 21;
  nRhoBins_ = 45 ;

}

// ADD map destructor
QGLikelihoodCalculator::~QGLikelihoodCalculator()
{
}



float QGLikelihoodCalculator::computeQGLikelihoodPU( float pt, float rhoPF, int nCharged, int nNeutral, float ptD, float rmsCand ) {


  double ptBins[100];
  Bins::getBins_int( Bins::nPtBins+1, ptBins, Bins::Pt0,Bins::Pt1,true);
  ptBins[Bins::nPtBins+1]=Bins::PtLastExtend;

  double rhoBins[100];
  Bins::getBins_int(Bins::nRhoBins+1,rhoBins,Bins::Rho0,Bins::Rho1,false);

  double ptMin=0.;
  double ptMax=0;
  double rhoMin=0.;
  double rhoMax=0;
  
  if(Bins::getBin(Bins::nPtBins,ptBins,pt,&ptMin,&ptMax) <0 ) return -1;
  if(Bins::getBin(Bins::nRhoBins,rhoBins,rhoPF,&rhoMin,&rhoMax) <0 ) return -1;

  int rhoBin = rhoMin;

  if( ptMax==0. ) return -1.;



  char histoName[300];
  sprintf( histoName, "rhoBins_pt%.0f_%.0f/nChargedJet0_gluon_pt%.0f_%.0f_rho%d", ptMin, ptMax, ptMin, ptMax, rhoBin);
	if(plots_[histoName]==NULL)
		plots_[histoName]=(TH1F*)histoFile_->Get(histoName)->Clone();
  	TH1F* h1_nCharged_gluon = plots_[histoName];
  sprintf( histoName, "rhoBins_pt%.0f_%.0f/nChargedJet0_quark_pt%.0f_%.0f_rho%d", ptMin, ptMax, ptMin, ptMax, rhoBin);
	if(plots_[histoName]==NULL)
		plots_[histoName]=(TH1F*)histoFile_->Get(histoName)->Clone();
  	TH1F* h1_nCharged_quark = plots_[histoName];

  sprintf( histoName, "rhoBins_pt%.0f_%.0f/nNeutralJet0_gluon_pt%.0f_%.0f_rho%d", ptMin, ptMax, ptMin, ptMax, rhoBin);
	if(plots_[histoName]==NULL)
		plots_[histoName]=(TH1F*)histoFile_->Get(histoName)->Clone();
  	TH1F* h1_nNeutral_gluon = plots_[histoName];

  sprintf( histoName, "rhoBins_pt%.0f_%.0f/nNeutralJet0_quark_pt%.0f_%.0f_rho%d", ptMin, ptMax, ptMin, ptMax, rhoBin);
	if(plots_[histoName]==NULL)
		plots_[histoName]=(TH1F*)histoFile_->Get(histoName)->Clone();
  	TH1F* h1_nNeutral_quark = plots_[histoName];

  sprintf( histoName, "rhoBins_pt%.0f_%.0f/ptDJet0_gluon_pt%.0f_%.0f_rho%d", ptMin, ptMax, ptMin, ptMax, rhoBin);
	if(plots_[histoName]==NULL && ptD>=0.)
		plots_[histoName]=(TH1F*)histoFile_->Get(histoName)->Clone();
  	TH1F* h1_ptD_gluon = (ptD>=0.) ? plots_[histoName] : 0;
  sprintf( histoName, "rhoBins_pt%.0f_%.0f/ptDJet0_quark_pt%.0f_%.0f_rho%d", ptMin, ptMax, ptMin, ptMax, rhoBin);
	if(plots_[histoName]==NULL && ptD>=0.)
	plots_[histoName]=(TH1F*)histoFile_->Get(histoName)->Clone();
  	TH1F* h1_ptD_quark = (ptD>=0.) ? plots_[histoName] : 0;

  sprintf( histoName, "rhoBins_pt%.0f_%.0f/rmsCandJet0_gluon_pt%.0f_%.0f_rho%d", ptMin, ptMax, ptMin, ptMax, rhoBin);
	if(plots_[histoName]==NULL && rmsCand>=0.)
	plots_[histoName]=(TH1F*)histoFile_->Get(histoName)->Clone();
  	TH1F* h1_rmsCand_gluon = (rmsCand>=0.) ? plots_[histoName] : 0;
  sprintf( histoName, "rhoBins_pt%.0f_%.0f/rmsCandJet0_quark_pt%.0f_%.0f_rho%d", ptMin, ptMax, ptMin, ptMax, rhoBin);
	if(plots_[histoName]==NULL && rmsCand>=0.)
	plots_[histoName]=(TH1F*)histoFile_->Get(histoName)->Clone();
  	TH1F* h1_rmsCand_quark = (rmsCand>=0.) ? plots_[histoName]: 0;


  float gluonP = likelihoodProduct( nCharged, nNeutral, ptD, rmsCand, h1_nCharged_gluon, h1_nNeutral_gluon, h1_ptD_gluon, h1_rmsCand_gluon );
  float quarkP = likelihoodProduct( nCharged, nNeutral, ptD, rmsCand, h1_nCharged_quark, h1_nNeutral_quark, h1_ptD_quark, h1_rmsCand_quark );

  float QGLikelihood = quarkP / (gluonP + quarkP );

 // if(h1_nCharged_gluon) delete h1_nCharged_gluon;
 // if(h1_nCharged_quark) delete h1_nCharged_quark;
 // if(h1_nNeutral_gluon) delete h1_nNeutral_gluon;
 // if(h1_nNeutral_quark) delete h1_nNeutral_quark;
 // if(h1_ptD_gluon) delete h1_ptD_gluon;
 // if(h1_ptD_quark) delete h1_ptD_quark;
 // if(h1_rmsCand_gluon) delete h1_rmsCand_gluon;
 // if(h1_rmsCand_quark) delete h1_rmsCand_quark;

  return QGLikelihood;

}



//new
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



	#ifdef DEBUG
	fprintf(stderr,"computeQG\n");
	#endif
double RhoBins[100];
double PtBins[100];

	#ifdef DEBUG
	fprintf(stderr,"computeBins\n");
	#endif
Bins::getBins_int(Bins::nPtBins+1,PtBins,Bins::Pt0,Bins::Pt1,true);
PtBins[Bins::nPtBins+1]=Bins::PtLastExtend;
Bins::getBins_int(Bins::nRhoBins+1,RhoBins,Bins::Rho0,Bins::Rho1,false);


double ptMin=0.;
double ptMax=0;
double rhoMin=0.;
double rhoMax=0;

if(Bins::getBin(Bins::nPtBins,PtBins,pt,&ptMin,&ptMax) <0 ) return -1;
if(Bins::getBin(Bins::nRhoBins,RhoBins,rho,&rhoMin,&rhoMax) <0 ) return -1;
//get Histo

float Q=1;
float G=1;
char histoName[1023];
ptMin=ceil(ptMin);
ptMax=ceil(ptMax);
rhoMin=floor(rhoMin);
rhoMax=floor(rhoMax);
	#ifdef DEBUG
	fprintf(stderr,"Start LOOP %.0f %.0f %.0f %.0f\n",ptMin,ptMax,rhoMin,rhoMax);
	#endif

for(unsigned int i=0;i<vars.size();i++){
//get Histo
	float Qi(1),Gi(1),mQ,mG;
	#ifdef DEBUG
	fprintf(stderr,"var %d = %s, value = %f\n",i,varName[i].c_str(), vars[i]);
	#endif
  	sprintf( histoName, "rhoBins_pt%.0f_%.0f/%s_quark_pt%.0f_%.0f_rho%.0f", ptMin, ptMax,varName[i].c_str(), ptMin, ptMax, rhoMin);
	#ifdef DEBUG
	fprintf(stderr,"looking for histo: %s\n", histoName );
	#endif
	if( plots_[histoName] == NULL ) { //first time 
        plots_[histoName]=(TH1F*)histoFile_->Get(histoName); 
        if( plots_[histoName]->GetEntries()<50 ) plots_[histoName]->Rebin(5); // try to make it more stable
        else if( plots_[histoName]->GetEntries()<500 ) plots_[histoName]->Rebin(2); // try to make it more stable
      }
	if( plots_[histoName] == NULL ) fprintf(stderr,"Histo %s does not exists\n",histoName); //DEBUG
	plots_[ histoName]->Scale(1./plots_[histoName]->Integral("width")); 

	Qi=plots_[histoName]->GetBinContent(plots_[histoName]->FindBin(vars[i]));
	mQ=plots_[histoName]->GetMean();
	
  	sprintf( histoName, "rhoBins_pt%.0f_%.0f/%s_gluon_pt%.0f_%.0f_rho%.0f", ptMin, ptMax,varName[i].c_str(), ptMin, ptMax, rhoMin);
	#ifdef DEBUG
	fprintf(stderr,"looking for histo: %s\n", histoName );
	#endif
	if( plots_[histoName] == NULL ) { // first time
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
		if(mQ>mG)
			{if(vars[i]>mQ) {Qi=1-delta;Gi=delta;}
			else if(vars[i]<mG){Qi=delta;Gi=1-delta;}
			}
		else if(mQ<mG)
			{if(vars[i]<mQ) {Qi=1-delta;Gi=delta;}
			else if(vars[i]>mG){Qi=delta;Gi=1-delta;}
			}
	} 

	Q*=Qi;
	G*=Gi;	
	
	#ifdef DEBUG
	fprintf(stderr,"Q: %f\n",Q);
	#endif
	#ifdef DEBUG
	fprintf(stderr,"G: %f\n",G);
	#endif
	}

if(Q==0) return 0;
return Q/(Q+G);

}


float QGLikelihoodCalculator::likelihoodProduct( float nCharged, float nNeutral, float ptD, float rmsCand, TH1F* h1_nCharged, TH1F* h1_nNeutral, TH1F* h1_ptD, TH1F* h1_rmsCand) {

  h1_nCharged->Scale(1./h1_nCharged->Integral("width"));
  if( h1_nNeutral!=0 )
    h1_nNeutral->Scale(1./h1_nNeutral->Integral("width"));
  if( h1_ptD!=0 )
    h1_ptD->Scale(1./h1_ptD->Integral("width"));
  if( h1_rmsCand!=0 )
    h1_rmsCand->Scale(1./h1_rmsCand->Integral("width"));


  float likeliProd =  h1_nCharged->GetBinContent(h1_nCharged->FindBin(nCharged));
  if( h1_nNeutral!=0 )
    likeliProd*=h1_nNeutral->GetBinContent(h1_nNeutral->FindBin(nNeutral));
  if( h1_ptD!=0 )
    likeliProd*=h1_ptD->GetBinContent(h1_ptD->FindBin(ptD));
  if( h1_rmsCand!=0 )
    likeliProd*=h1_rmsCand->GetBinContent(h1_rmsCand->FindBin(rmsCand));


  return likeliProd;

}


Float_t QGLikelihoodCalculator::QGvalue(std::map<TString, Float_t> variables){
  return computeQGLikelihood2012(variables["pt"], variables["eta"], variables["rhoIso"], variables["mult"], variables["ptD"], variables["axis2"]); 
}

