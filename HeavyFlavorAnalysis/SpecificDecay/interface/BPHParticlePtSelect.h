#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHParticlePtSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHParticlePtSelect_h
/** \class BPHParticlePtSelect
 *
 *  Description: 
 *     Class for particle selection by Pt
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHParticlePtSelect : public BPHRecoSelect {
public:
  /** Constructor
   */
  BPHParticlePtSelect(double pt) : ptMin(pt) {}

  // deleted copy constructor and assignment operator
  BPHParticlePtSelect(const BPHParticlePtSelect& x) = delete;
  BPHParticlePtSelect& operator=(const BPHParticlePtSelect& x) = delete;

  /** Destructor
   */
  ~BPHParticlePtSelect() override = default;

  /** Operations
   */
  /// select particle
  bool accept(const reco::Candidate& cand) const override { return (cand.p4().pt() >= ptMin); }

  /// set pt min
  void setPtMin(double pt) {
    ptMin = pt;
    return;
  }

  /// get current pt min
  double getPtMin() const { return ptMin; }

private:
  double ptMin;
};

#endif
