#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHKinFitChi2Select_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHKinFitChi2Select_h
/** \class BPHKinFitChi2Select
 *
 *  Description: 
 *     Class for candidate selection by chisquare (at vertex fit level)
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHFitSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHKinematicFit.h"
#include "TMath.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHKinFitChi2Select : public BPHFitSelect {
public:
  /** Constructor
   */
  BPHKinFitChi2Select(double prob) : probMin(prob) {}

  // deleted copy constructor and assignment operator
  BPHKinFitChi2Select(const BPHKinFitChi2Select& x) = delete;
  BPHKinFitChi2Select& operator=(const BPHKinFitChi2Select& x) = delete;

  /** Destructor
   */
  ~BPHKinFitChi2Select() override = default;

  /** Operations
   */
  /// select vertex
  bool accept(const BPHKinematicFit& cand) const override {
    if (probMin < 0.0)
      return true;
    const RefCountedKinematicVertex tdv = cand.topDecayVertex();
    if (tdv.get() == nullptr)
      return false;
    if (!tdv->vertexIsValid())
      return false;
    reco::Vertex v(*tdv);
    if (v.isFake())
      return false;
    if (!v.isValid())
      return false;
    return (TMath::Prob(v.chi2(), lround(v.ndof())) >= probMin);
  }

  /// set prob min
  void setProbMin(double p) {
    probMin = p;
    return;
  }

  /// get current prob min
  double getProbMin() const { return probMin; }

private:
  double probMin;
};

#endif
