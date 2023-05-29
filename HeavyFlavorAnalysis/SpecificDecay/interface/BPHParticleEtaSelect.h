#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHParticleEtaSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHParticleEtaSelect_h
/** \class BPHParticleEtaSelect
 *
 *  Descrietaion: 
 *     Class for particle selection by eta
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

class BPHParticleEtaSelect : public BPHRecoSelect {
public:
  /** Constructor
   */
  BPHParticleEtaSelect(double eta) : etaMax(eta) {}

  // deleted copy constructor and assignment operator
  BPHParticleEtaSelect(const BPHParticleEtaSelect& x) = delete;
  BPHParticleEtaSelect& operator=(const BPHParticleEtaSelect& x) = delete;

  /** Destructor
   */
  ~BPHParticleEtaSelect() override = default;

  /** Operations
   */
  /// select particle
  bool accept(const reco::Candidate& cand) const override { return (fabs(cand.p4().eta()) <= etaMax); }

  /// set eta max
  void setEtaMax(double eta) {
    etaMax = eta;
    return;
  }

  /// get current eta max
  double getEtaMax() const { return etaMax; }

private:
  double etaMax;
};

#endif
