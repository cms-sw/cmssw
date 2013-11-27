/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#ifndef ChiSquareFunctionUpdator_h
#define ChiSquareFunctionUpdator_h

#include "Minuit2/FCNBase.h"
#include "TMatrixT.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"

class ChiSquareFunctionUpdator  : public ROOT::Minuit2::FCNBase {
 public:
  ChiSquareFunctionUpdator(TrackHelixVertexFitter *VF_){VF=VF_;}
  virtual ~ChiSquareFunctionUpdator(){};
  
  virtual double operator() (const std::vector<double> & x)const{
    TMatrixT<double> X(x.size(),1);
    for(unsigned int i=0; i<x.size();i++){X(i,0)=x.at(i);}
    return VF->UpdateChisquare(X);
  }
  virtual double Up()const{return 1.0;}// Error definiton for Chi^2

 private:
  TrackHelixVertexFitter *VF;
  
};
#endif

