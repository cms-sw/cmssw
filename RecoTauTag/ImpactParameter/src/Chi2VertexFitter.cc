/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/ImpactParameter/interface/Chi2VertexFitter.h"
#include "RecoTauTag/ImpactParameter/interface/ChiSquareFunctionUpdator.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnSimplex.h"
#include "Minuit2/CombinedMinimizer.h"
#include "Minuit2/MnMinimize.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MnContours.h"
#include "Minuit2/MnPlot.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/ContoursError.h"
#include <iostream>

using namespace tauImpactParameter;

bool Chi2VertexFitter::fit(){
  if(isFit_==true) return true;// do not refit
  if(!isConfigured_) return false; // do not fit if configuration failed
  ChiSquareFunctionUpdator updator(this);
  ROOT::Minuit2::MnUserParameters MnPar;
  for(int i=0;i<par_.GetNrows();i++){
    TString name=freeParName(i);
    // if not limited (vhigh <= vlow)
    MnPar.Add(name.Data(),par_(i),sqrt(fabs(parcov_(i,i))),par_(i)-nsigma_*sqrt(fabs(parcov_(i,i))),par_(i)+nsigma_*sqrt(fabs(parcov_(i,i))));
  }

  unsigned int max=10;
  int numberofcalls=200+par_.GetNrows()*100+par_.GetNrows()*par_.GetNrows()*5;
  double tolerance(0.01);
  double edmMin(0.001*updator.Up()*tolerance); 

  ROOT::Minuit2::MnMinimize minimize(updator,MnPar);
  ROOT::Minuit2::FunctionMinimum min= minimize(numberofcalls,tolerance);
  for(unsigned int i=0;i<=max && min.Edm()>edmMin;i++){
    if(i==max) return false;
    min = minimize(i*numberofcalls,tolerance);
  }
  // give return flag based on status
  if(min.IsAboveMaxEdm()){edm::LogWarning("Chi2VertexFitter::Fit") << "Found Vertex that is above EDM " << std::endl; return false;}
  if(!min.IsValid()){
    edm::LogWarning("Chi2VertexFitter::Fit") << "Failed min.IsValid()" << std::endl; 
    if(!min.HasValidParameters()){edm::LogWarning("Chi2VertexFitter::Fit") << "Failed min.HasValidParameters()" << std::endl; }
    if(!min.HasValidCovariance()){edm::LogWarning("Chi2VertexFitter::Fit") << "Failed min.HasValidCovariance()" << std::endl; }
    if(!min.HesseFailed()){edm::LogWarning("Chi2VertexFitter::Fit") << "Failed min.HesseFailed()" << std::endl; }
    if(!min.HasReachedCallLimit()){edm::LogWarning("Chi2VertexFitter::Fit") << "Failed min.HasReachedCallLimit()" << std::endl; }
    return false;
  }
  chi2_=min.Fval();
  // Get output parameters
  for(int i=0;i<par_.GetNrows();i++){ par_(i)=min.UserParameters().Value(i);}
  // Get output covariance
  for(int i=0;i<par_.GetNrows();i++){
    for(int j=0;j<par_.GetNrows();j++){
      parcov_(i,j)=min.UserCovariance()(i,j);
    }
  }

  isFit_=true;
  return isFit_;
}
