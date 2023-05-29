#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHMuonChargeSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHMuonChargeSelect_h
/** \class BPHMuonChargeSelect
 *
 *  Description: 
 *     Class for muon selection by charge
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleChargeSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/PatCandidates/interface/Muon.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHMuonChargeSelect : public BPHParticleChargeSelect {
public:
  /** Constructor
   */
  BPHMuonChargeSelect(int c) : BPHParticleChargeSelect(c) {}

  // deleted copy constructor and assignment operator
  BPHMuonChargeSelect(const BPHMuonChargeSelect& x) = delete;
  BPHMuonChargeSelect& operator=(const BPHMuonChargeSelect& x) = delete;

  /** Destructor
   */
  ~BPHMuonChargeSelect() override = default;

  /** Operations
   */
  /// select muon
  bool accept(const reco::Candidate& cand) const override {
    if (dynamic_cast<const pat::Muon*>(&cand) == nullptr)
      return false;
    return BPHParticleChargeSelect::accept(cand);
  };
};

#endif
