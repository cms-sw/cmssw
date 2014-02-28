#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"
#include "RecoJets/JetAlgorithms/interface/Bins.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

#include "TMath.h"
#include "TH1F.h"

using namespace std;


QGLikelihoodCalculator::QGLikelihoodCalculator( TString dataDir, bool chs){

  if( !(dataDir.EndsWith("/")) ) dataDir += "/";

  TString histoFileName = "ReducedHisto_2012.root";
  if(chs) histoFileName = "ReducedHisto_2012_CHS.root";
  histoFile = TFile::Open(TString(edm::FileInPath(dataDir + histoFileName).fullPath()));

  Bins::getBins_int(RhoBins, Bins::nRhoBins, Bins::Rho0, Bins::Rho1, false);
  Bins::getBins_int(PtBins, Bins::nPtBins, Bins::Pt0, Bins::Pt1, true);
  PtBins.push_back(4000);

  for(int i = 0; i < 2*2*3*Bins::nPtBins*Bins::nRhoBins; ++i) plots.push_back(nullptr);
}

int QGLikelihoodCalculator::indexTH1F(int etaIndex, int qgIndex, int varIndex, int ptIndex, int rhoIndex){
  return (((etaIndex + 2*qgIndex)*3 + varIndex)*Bins::nPtBins + ptIndex)*Bins::nRhoBins + rhoIndex;
}

void QGLikelihoodCalculator::loadTH1F(int etaIndex, int qgIndex, int varIndex, int ptIndex, int rhoIndex){
  int index = indexTH1F(etaIndex, qgIndex, varIndex, ptIndex, rhoIndex);
  TString ptMin = TString::Format("%d", PtBins.at(ptIndex));
  TString ptMax = TString::Format("%d", PtBins.at(ptIndex+1));
  TString rhoMin = TString::Format("%d", RhoBins.at(rhoIndex));
  TString varName;
  if(varIndex == 0) varName = "nPFCand_QC_ptCutJet0";
  if(varIndex == 1) varName = "ptD_QCJet0";
  if(varIndex == 2) varName = "axis2_QCJet0";
  TString histoName = "rhoBins_pt" + ptMin + "_" + ptMax + "/" + varName + (etaIndex==0?"":"_F") + "_" + (qgIndex==0?"quark":"gluon") + "_pt" + ptMin + "_" + ptMax + "_rho" + rhoMin;
  histoFile->GetObject(histoName, plots.at(index)); 
  if(!plots.at(index)){ std::cout << "Error (RecoJets/JetAlgorithms/QGLikelihoodCalculator.cc): " << histoName << " not found!" << std::endl; exit(1);}
  if(plots.at(index)->GetEntries()<50 ) 	plots.at(index)->Rebin(5); 	// try to make it more stable
  else if(plots.at(index)->GetEntries()<500 ) 	plots.at(index)->Rebin(2); 	// try to make it more stable
  plots.at(index)->Scale(1./plots.at(index)->Integral("width")); 
}


float QGLikelihoodCalculator::computeQGLikelihood2012(float pt, float eta, float rho, int nPFCandidates_QC_ptCut, float ptD_QC, float axis2_QC){

  int etaIndex = (fabs(eta)>2.5);
  if(etaIndex && pt>127.) pt = 128.;		// in forward use inclusive 127-4000 bin

  int ptBin = Bins::getBinNumber(PtBins, pt);
  if(ptBin == -1) return -1;
  int rhoBin = Bins::getBinNumber(RhoBins, rho);
  if(rhoBin == -1) return -1;

  std::vector<float> vars;
  vars.push_back(nPFCandidates_QC_ptCut);
  vars.push_back(ptD_QC);
  vars.push_back(-log(axis2_QC)); //-log

  float Q=1;
  float G=1;

  for(unsigned int varIndex = 0; varIndex < 3; ++varIndex){
    int index = indexTH1F(etaIndex, 0, varIndex, ptBin, rhoBin);
    if(!plots.at(index)) loadTH1F(etaIndex, 0, varIndex, ptBin, rhoBin); 

    float Qi = plots.at(index)->GetBinContent(plots.at(index)->FindBin(vars[varIndex]));
    float mQ = plots.at(index)->GetMean();
	
    index = indexTH1F(etaIndex, 1, varIndex, ptBin, rhoBin);
    if(!plots.at(index)) loadTH1F(etaIndex, 1, varIndex, ptBin, rhoBin); 

    float Gi = plots.at(index)->GetBinContent(plots.at(index)->FindBin(vars[varIndex]));
    float mG = plots.at(index)->GetMean();

	
    float epsilon=0;
    float delta=0.000001;
    if(Qi <= epsilon && Gi <= epsilon){
      if(mQ>mG){
	if(vars[varIndex] > mQ){ Qi = 1-delta; Gi = delta;}
	else if(vars[varIndex] < mG){ Qi = delta; Gi = 1-delta;}
      }
      else if(mQ<mG){
	if(vars[varIndex]<mQ) { Qi = 1-delta; Gi = delta;}
	else if(vars[varIndex]>mG){Qi = delta;Gi = 1-delta;}
      }
    } 

    Q*=Qi;
    G*=Gi;	
  }

  if(Q==0) return 0;
  return Q/(Q+G);
}
