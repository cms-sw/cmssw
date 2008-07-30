#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SignAlgoResolutions
// 
/**\class METSignificance SignAlgoResolutions.cc RecoMET/METAlgorithms/src/SignAlgoResolutions.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kyle Story, Freya Blekman (Cornell University)
//         Created:  Fri Apr 18 11:58:33 CEST 2008
// $Id: SignAlgoResolutions.cc,v 1.1 2008/04/18 10:12:55 fblekman Exp $
//
//
#include "FWCore/Framework/interface/EventSetup.h"
#include <math.h>


double metsig::SignAlgoResolutions::eval(const resolutionType & type, const resolutionFunc & func, const double & et, const double & phi, const double & eta) const {

  functionPars x(3);
  x[0]=et;
  x[1]=phi;
  x[2]=eta;
  
  return getfunc(type,func,x);

}
metsig::SignAlgoResolutions::SignAlgoResolutions(const edm::ParameterSet &iConfig):functionmap_(){
  addResolutions(iConfig);
}

void metsig::SignAlgoResolutions::addResolutions(const edm::ParameterSet &iConfig){
  // for now: do this by hand:
  functionPars etparameters(3,0);
  functionPars phiparameters(1,0);
  // set the parameters per function:
  // ECAL, BARREL:
  std::vector<double> ebet = iConfig.getParameter<std::vector<double> >("EB_EtResPar");
  std::vector<double> ebphi = iConfig.getParameter<std::vector<double> >("EB_PhiResPar");
  std::cout << ebet.size() << " " << ebphi.size() << std::endl;
  etparameters[0]=ebet[0];
  etparameters[1]=ebet[1];
  etparameters[2]=ebet[2];
  phiparameters[0]=ebphi[0];
  addfunction(caloEB,ET,etparameters);
  addfunction(caloEB,PHI,phiparameters);
 // ECAL, ENDCAP:
  std::vector<double> eeet = iConfig.getParameter<std::vector<double> >("EE_EtResPar");
  std::vector<double> eephi = iConfig.getParameter<std::vector<double> >("EE_PhiResPar");
  etparameters[0]=eeet[0];
  etparameters[1]=eeet[1];
  etparameters[2]=eeet[2];
  phiparameters[0]=eephi[0];
  addfunction(caloEE,ET,etparameters);
  addfunction(caloEE,PHI,phiparameters);
 // HCAL, BARREL:
  std::vector<double> hbet = iConfig.getParameter<std::vector<double> >("HB_EtResPar");
  std::vector<double> hbphi = iConfig.getParameter<std::vector<double> >("HB_PhiResPar");
  etparameters[0]=hbet[0];
  etparameters[1]=hbet[1];
  etparameters[2]=hbet[2];
  phiparameters[0]=hbphi[0];
  addfunction(caloHB,ET,etparameters);
  addfunction(caloHB,PHI,phiparameters);
 // HCAL, ENDCAP:
  std::vector<double> heet = iConfig.getParameter<std::vector<double> >("HE_EtResPar");
  std::vector<double> hephi = iConfig.getParameter<std::vector<double> >("HE_PhiResPar");
  etparameters[0]=heet[0];
  etparameters[1]=heet[1];
  etparameters[2]=heet[2];
  phiparameters[0]=hephi[0];
  addfunction(caloHE,ET,etparameters);
  addfunction(caloHE,PHI,phiparameters);
 // HCAL, Outer
  std::vector<double> hoet = iConfig.getParameter<std::vector<double> >("HO_EtResPar");
  std::vector<double> hophi = iConfig.getParameter<std::vector<double> >("HO_PhiResPar");
  etparameters[0]=hoet[0];
  etparameters[1]=hoet[1];
  etparameters[2]=hoet[2];
  phiparameters[0]=hophi[0];
  addfunction(caloHO,ET,etparameters);
  addfunction(caloHO,PHI,phiparameters);
 // HCAL, Forward
  std::vector<double> hfet = iConfig.getParameter<std::vector<double> >("HF_EtResPar");
  std::vector<double> hfphi = iConfig.getParameter<std::vector<double> >("HF_PhiResPar");
  etparameters[0]=hfet[0];
  etparameters[1]=hfet[1];
  etparameters[2]=hfet[2];
  phiparameters[0]=hfphi[0];
  addfunction(caloHF,ET,etparameters);
  addfunction(caloHF,PHI,phiparameters);

  return;
}

void metsig::SignAlgoResolutions::addfunction(resolutionType type, resolutionFunc func, functionPars parameters){

  //  std::cout << "adding function for " << type << " " << func << ", parameters " ;
  //  for(size_t ii=0; ii<parameters.size();++ii)
  //    std::cout << parameters[ii] << " ";
  //  std::cout << std::endl;
  functionCombo mypair(type,func);
  functionmap_[mypair]=parameters;
  
}

double metsig::SignAlgoResolutions::getfunc(const metsig::resolutionType & type,const metsig::resolutionFunc & func, functionPars & x) const{
  
  double result=0;
  functionCombo mypair(type,func);
  
 
  
  if(functionmap_.count(mypair)==0)
    return result;
  
  functionPars values = (functionmap_.find(mypair))->second;
  if(func==metsig::ET)
    result = EtFunction(x,values);
  else if(func==metsig::PHI)
    result = PhiFunction(x,values);
  
  // std::cout << "returning function " << type << " " << func << " " << result << " " << x[0] << std::endl; 

  return result;
}

double metsig::SignAlgoResolutions::EtFunction( const functionPars &x, const functionPars & par) const
{
  if(par.size()<3)
    return 0.;
  if(x.size()<1)
    return 0.;
  double et=x[0];
  if(et<=0.)
    return 0.;
  double result = et*sqrt((par[2]*par[2])+(par[1]*par[1]/et)+(par[0]*par[0]/(et*et)));
  return result;
}


double metsig::SignAlgoResolutions::PhiFunction(const functionPars &x,const  functionPars & par) const
{
  double et=x[0];
  if(et<=0.)
    return 0.;
  double result = par[0]*et;
  return result;

}
