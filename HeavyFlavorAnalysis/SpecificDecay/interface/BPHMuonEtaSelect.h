#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHMuonEtaSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHMuonEtaSelect_h
/** \class BPHMuonEtaSelect
 *
 *  Descrietaion: 
 *     Class for muon selection by eta
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"

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

class BPHMuonEtaSelect : public BPHParticleEtaSelect {
public:
  /** Constructor
   */
  BPHMuonEtaSelect(double eta) : BPHParticleEtaSelect(eta) {}

  // deleted copy constructor and assignment operator
  BPHMuonEtaSelect(const BPHMuonEtaSelect& x) = delete;
  BPHMuonEtaSelect& operator=(const BPHMuonEtaSelect& x) = delete;

  /** Destructor
   */
  ~BPHMuonEtaSelect() override = default;

  /** Operations
   */
  /// select muon
  bool accept(const reco::Candidate& cand) const override {
    if (dynamic_cast<const pat::Muon*>(&cand) == nullptr)
      return false;
    return BPHParticleEtaSelect::accept(cand);
  }
};

#endif
