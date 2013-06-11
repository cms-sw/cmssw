/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
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

bool Chi2VertexFitter::Fit(){
  if(isFit==true) return true;// do not refit
  if(!isConfigure) return false; // do not fit if configuration failed
  ChiSquareFunctionUpdator updator(this);
  ROOT::Minuit2::MnUserParameters MnPar;
  for(int i=0;i<par.GetNrows();i++){
    TString name=FreeParName(i);
    // if not limited (vhigh <= vlow)
    MnPar.Add(name.Data(),par(i,0),sqrt(fabs(parcov(i,i))),par(i,0)-nsigma*sqrt(fabs(parcov(i,i))),par(i,0)+nsigma*sqrt(fabs(parcov(i,i))));
  }

  unsigned int max=10;
  int numberofcalls=200+par.GetNrows()*100+par.GetNrows()*par.GetNrows()*5;
  double tolerance(0.01);
  double edmMin(0.001*updator.Up()*tolerance); 

  ROOT::Minuit2::MnMinimize minimize(updator,MnPar);
  ROOT::Minuit2::FunctionMinimum min= minimize(numberofcalls,tolerance);
  for(unsigned int i=0;i<=max && min.Edm()>edmMin;i++){
    if(i==max) return false;
    min = minimize(i*numberofcalls,tolerance);
  }
  // give return flag based on status
  if(min.IsAboveMaxEdm()){std::cout << "Found Vertex that is above EDM " << std::endl; return false;}
  if(!min.IsValid()){
    std::cout << "Chi2VertexFitter::Fit(): Failed min.IsValid()" << std::endl; 
    if(!min.HasValidParameters()){std::cout << "Chi2VertexFitter::Fit(): Failed min.HasValidParameters()" << std::endl; }
    if(!min.HasValidCovariance()){std::cout << "Chi2VertexFitter::Fit(): Failed min.HasValidCovariance()" << std::endl; }
    if(!min.HesseFailed()){std::cout << "Chi2VertexFitter::Fit(): Failed min.HesseFailed()" << std::endl; }
    if(!min.HasReachedCallLimit()){std::cout << "Chi2VertexFitter::Fit(): Failed min.HasReachedCallLimit()" << std::endl; }
    return false;
  }
  chi2=min.Fval();
  // Get output parameters
  for(int i=0;i<par.GetNrows();i++){ par(i,0)=min.UserParameters().Value(i);}
  // Get output covariance
  for(int i=0;i<par.GetNrows();i++){
    for(int j=0;j<par.GetNrows();j++){parcov(i,j)=min.UserCovariance()(i,j);}
  }

  isFit=true;
  return isFit;
}
