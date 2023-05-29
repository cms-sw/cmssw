#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHParticleChargeSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHParticleChargeSelect_h
/** \class BPHParticleChargeSelect
 *
 *  Description: 
 *     Class for particle selection by charge
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

class BPHParticleChargeSelect : public BPHRecoSelect {
public:
  /** Constructor
   */
  BPHParticleChargeSelect(int c) : charge(c ? (c > 0 ? 1 : -1) : 0) {}

  // deleted copy constructor and assignment operator
  BPHParticleChargeSelect(const BPHParticleChargeSelect& x) = delete;
  BPHParticleChargeSelect& operator=(const BPHParticleChargeSelect& x) = delete;

  /** Destructor
   */
  ~BPHParticleChargeSelect() override = default;

  /** Operations
   */
  /// select particle
  bool accept(const reco::Candidate& cand) const override {
    switch (charge) {
      default:
      case 0:
        return (cand.charge() != 0);
      case 1:
        return (cand.charge() > 0);
      case -1:
        return (cand.charge() < 0);
    }
    return true;
  };

  /// set selection charge
  void setCharge(int c) {
    charge = (c ? (c > 0 ? 1 : -1) : 0);
    return;
  }

  /// get selection charge
  double getCharge() const { return charge; }

private:
  int charge;
};

#endif
