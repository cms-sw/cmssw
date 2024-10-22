#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHChi2Select_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHChi2Select_h
/** \class BPHChi2Select
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
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayVertex.h"
#include "TMath.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHChi2Select : public BPHVertexSelect {
public:
  /** Constructor
   */
  BPHChi2Select(double prob) : probMin(prob) {}

  // deleted copy constructor and assignment operator
  BPHChi2Select(const BPHChi2Select& x) = delete;
  BPHChi2Select& operator=(const BPHChi2Select& x) = delete;

  /** Destructor
   */
  ~BPHChi2Select() override = default;

  /** Operations
   */
  /// select vertex
  bool accept(const BPHDecayVertex& cand) const override {
    if (probMin < 0.0)
      return true;
    const reco::Vertex& v = cand.vertex();
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
