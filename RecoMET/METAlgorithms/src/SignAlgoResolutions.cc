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
// $Id$
//
//
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
  etparameters[0]=0.2;
  etparameters[1]=0.03;
  etparameters[2]=0.005;
  phiparameters[0]=0.0174;
  addfunction(caloEB,ET,etparameters);
  addfunction(caloEB,PHI,phiparameters);
 // ECAL, ENDCAP:
  etparameters[0]=0.2;
  etparameters[1]=0.03;
  etparameters[2]=0.005;
  phiparameters[0]=0.087;
  addfunction(caloEE,ET,etparameters);
  addfunction(caloEE,PHI,phiparameters);
 // HCAL, BARREL:
  etparameters[0]=0.;
  etparameters[1]=1.22;
  etparameters[2]=0.05;
  phiparameters[0]=0.087;
  addfunction(caloHB,ET,etparameters);
  addfunction(caloHB,PHI,phiparameters);
 // HCAL, ENDCAP:
  etparameters[0]=0.;
  etparameters[1]=1.3;
  etparameters[2]=0.05;
  phiparameters[0]=0.087;
  addfunction(caloHE,ET,etparameters);
  addfunction(caloHE,PHI,phiparameters);
 // HCAL, Outer
  etparameters[0]=0.;
  etparameters[1]=1.3;
  etparameters[2]=0.005;
  phiparameters[0]=0.087;
  addfunction(caloHO,ET,etparameters);
  addfunction(caloHO,PHI,phiparameters);
 // HCAL, Forward
  etparameters[0]=0.;
  etparameters[1]=1.82;
  etparameters[2]=0.09;
  phiparameters[0]=0.174;
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
