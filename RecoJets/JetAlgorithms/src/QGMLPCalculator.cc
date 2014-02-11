#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TMath.h"
#include "RecoJets/JetAlgorithms/interface/QGMLPCalculator.h"



QGMLPCalculator::QGMLPCalculator(const TString mva_, const TString dataDir, const Bool_t useProbValue_){
  mva = mva_;
  useProbValue = useProbValue_;

  TString variableNames[] = {"axis1","axis2","mult","ptD"};
  reader = new TMVA::Reader("Silent");
  for(Int_t i=0; i < 4; ++i) reader->AddVariable(variableNames[i], &mvaVariables_corr[variableNames[i]]);
  for(Int_t i = 0; i <= this->getLastBin("c"); ++i) reader->BookMVA(mva+"c"+ str(i), edm::FileInPath(dataDir + "c" + str(i) + "_" + mva + ".xml").fullPath());
  for(Int_t i = 0; i <= this->getLastBin("f"); ++i) reader->BookMVA(mva+"f"+ str(i), edm::FileInPath(dataDir + "f" + str(i) + "_" + mva + ".xml").fullPath());

  setRhoCorrections();
}

Float_t QGMLPCalculator::interpolate(Double_t pt, Int_t ptlow, Int_t pthigh, Float_t &mvalow, Float_t &mvahigh){
  return (mvahigh-mvalow)/(pthigh-ptlow)*(pt-ptlow)+mvalow;
}


Float_t QGMLPCalculator::QGvalue(std::map<TString, Float_t> variables){
  //Define working region (no result if abs(eta) > 4.7)
  TString region = "c";
  if(abs(variables["eta"]) >=2.5) region = "f";
  if(abs(variables["eta"]) > 4.7) return -999;

  //Define pt bin
  if(variables["pt"] < getMinPt() || variables["pt"] > getMaxPt(region)) return -999;
  if(variables["pt"] < getBinsAveragePt(region , 0)) return QGvalueInBin(variables, region, 0); 				 //Below average lowest ptBin --> no interpolation
  if(variables["pt"] > getBinsAveragePt(region, getLastBin(region))) return QGvalueInBin(variables, region, getLastBin(region)); //Above average highest ptBin --> no interpolation

  Int_t ptBin = 0;
  while(variables["pt"] > getBinsAveragePt(region, ptBin)) ++ptBin; 
 
  //Calculate (interpolated) mva value
  Float_t QGvalueDown = QGvalueInBin(variables, region, ptBin-1);
  Float_t QGvalueUp = QGvalueInBin(variables, region, ptBin);
  return interpolate(variables["pt"], getBinsAveragePt(region, ptBin-1), getBinsAveragePt(region, ptBin), QGvalueDown, QGvalueUp);
}

Float_t QGMLPCalculator::QGvalueInBin(std::map<TString, Float_t> variables, TString region, Int_t ptBin){
  for(std::map<TString, Float_t>::iterator it = mvaVariables_corr.begin(); it != mvaVariables_corr.end(); ++it){
    mvaVariables_corr[it->first] = variables[it->first] - corrections[it->first + "_" + region + str(ptBin)]*variables["rho"];
  }
  if(useProbValue) return reader->GetProba(TString(mva) + region + str(ptBin));
  else             return reader->EvaluateMVA(TString(mva) + region + str(ptBin));
} 
