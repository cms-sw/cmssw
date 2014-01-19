#ifndef RecoTauTag_ImpactParameter_ChiSquareFunctionUpdator_h
#define RecoTauTag_ImpactParameter_ChiSquareFunctionUpdator_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "Minuit2/FCNBase.h"
#include "TMatrixT.h"
#include "TVectorT.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"

namespace tauImpactParameter {

class ChiSquareFunctionUpdator : public ROOT::Minuit2::FCNBase {
 public:
  ChiSquareFunctionUpdator(TrackHelixVertexFitter* VF){ VF_ = VF; }
  virtual ~ChiSquareFunctionUpdator() {};
  
  virtual double operator() (const std::vector<double>& x) const
  {
    TVectorT<double> X(x.size());
    for ( unsigned int i = 0; i < x.size(); ++i ) {
      X(i) = x[i];
    }
    return VF_->updateChisquare(X);
  }
  virtual double Up() const { return 1.0; }// Error definiton for Chi^2 (virtual function defined in ROOT::Minuit2::FCNBase base-class)

 private:
  TrackHelixVertexFitter* VF_; 
};

}
#endif

