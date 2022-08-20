#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHMassSymSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHMassSymSelect_h
/** \class BPHMassSymSelect
 *
 *  Description: 
 *     Class for candidate selection by invariant mass (at momentum sum level)
 *     allowing for decay product mass swap
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayMomentum.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHMassSymSelect : public BPHMomentumSelect {
public:
  /** Constructor
   */
  BPHMassSymSelect(const std::string& np, const std::string& nn, const BPHMassSelect* ms)
      : nPos(np), nNeg(nn), mSel(ms) {}

  // deleted copy constructor and assignment operator
  BPHMassSymSelect(const BPHMassSymSelect& x) = delete;
  BPHMassSymSelect& operator=(const BPHMassSymSelect& x) = delete;

  /** Destructor
   */
  ~BPHMassSymSelect() override = default;

  /** Operations
   */
  /// select particle
  bool accept(const BPHDecayMomentum& cand) const override {
    if (mSel->accept(cand))
      return true;

    const reco::Candidate* pp = cand.getDaug(nPos);
    const reco::Candidate* np = cand.getDaug(nNeg);

    reco::Candidate* pc = cand.originalReco(pp)->clone();
    reco::Candidate* nc = cand.originalReco(np)->clone();

    pc->setMass(np->p4().mass());
    nc->setMass(pp->p4().mass());
    const reco::Candidate::LorentzVector s4 = pc->p4() + nc->p4();
    double mass = s4.mass();

    delete pc;
    delete nc;
    return ((mass >= mSel->getMassMin()) && (mass <= mSel->getMassMax()));
  }

private:
  std::string nPos;
  std::string nNeg;
  const BPHMassSelect* mSel;
};

#endif
