#ifndef DataFormats_TauReco_PFTauDecayMode_h
#define DataFormats_TauReco_PFTauDecayMode_h

/* class PFTauDecayMode
 * 
 * Stores information for determing the type of hadronic decay of a tau lepton
 *
 * Associated to a reco::PFTau object
 * Provides functionality for:
 *      - merging gamma candidates into candidate PiZeroes
 *      - indexing reconstructed hadronic decay mode
 *      - computing vertex information for multi-prong
 *      - filtering suspected underlying event
 *                          
 * author: Evan K. Friis, UC Davis (evan.klose.friis@cern.ch)
 * created: Mon Jun 30 13:53:59 PDT 2008
 */

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeFwd.h"

namespace reco {
  class PFTauDecayMode : public CompositeCandidate {
  public:
    //to help with indexing.  changing this value necissitates changing the enum below
    const static unsigned char maxNumberOfPiZeroCandidatesAllowed = 4;

    enum hadronicTauDecayModes {
      tauDecay1ChargedPion0PiZero,
      tauDecay1ChargedPion1PiZero,  // rho (770 MeV) mediated)
      tauDecay1ChargedPion2PiZero,  // a1  (1.2 GeV) mediated
      tauDecay1ChargedPion3PiZero,  // contaminated or unmerged photo
      tauDecay1ChargedPion4PiZero,  // contaminated or unmerged photo
      tauDecay2ChargedPion0PiZero,  // extra track or un-recod track
      tauDecay2ChargedPion1PiZero,  // extra track or un-recod track
      tauDecay2ChargedPion2PiZero,  // extra track or un-recod track
      tauDecay2ChargedPion3PiZero,  // extra track or un-recod track
      tauDecay2ChargedPion4PiZero,  // extra track or un-recod track
      tauDecay3ChargedPion0PiZero,  // a1  (1.2 GeV) mediated
      tauDecay3ChargedPion1PiZero,  // a1  (1.2 GeV) mediated
      tauDecay3ChargedPion2PiZero,  // a1  (1.2 GeV) mediated
      tauDecay3ChargedPion3PiZero,  // a1  (1.2 GeV) mediated
      tauDecay3ChargedPion4PiZero,  // a1  (1.2 GeV) mediated
      tauDecaysElectron,
      tauDecayMuon,
      tauDecayOther  // catch-all
    };

    PFTauDecayMode() {}
    /// constructor from values
    PFTauDecayMode(Charge q,
                   const LorentzVector& p4,
                   const Point& vtx = Point(0, 0, 0),
                   int pdgId = 12,
                   int status = 2,
                   bool integerCharge = true)
        : CompositeCandidate(q, p4, vtx, pdgId, status, integerCharge) {}

    /// constructor from candidate content
    PFTauDecayMode(const VertexCompositeCandidate& chargedPions,
                   const CompositeCandidate& piZeroes,
                   const CompositeCandidate& filteredObjects);

    ~PFTauDecayMode() override {}
    PFTauDecayMode* clone() const override;

    /// return reference to associated PFTau object
    const PFTauRef& pfTauRef() const { return pfTauRef_; }
    void setPFTauRef(const PFTauRef& theTau) { pfTauRef_ = theTau; }

    hadronicTauDecayModes getDecayMode() const { return theDecayMode_; }
    void setDecayMode(hadronicTauDecayModes theDecayMode) { theDecayMode_ = theDecayMode; }

    /// returns collection of charged pions w/ vertex information (tracks are refit)
    const VertexCompositeCandidate& chargedPions() const;
    /// returns a collection of merged Pi0s
    const CompositeCandidate& neutralPions() const;
    /// returns references to PF objects that were filtered
    const CompositeCandidate& filteredObjects() const;

    /// returns pointers to charged pions
    std::vector<const Candidate*> chargedPionCandidates() const;
    /// returns pointers to neutral pions
    std::vector<const Candidate*> neutralPionCandidates() const;
    /// returns pointers to non-filtered objects
    std::vector<const Candidate*> decayProductCandidates() const;
    /// returns pointers to filtered objects (i.e. those not included in signal objects)
    std::vector<const Candidate*> filteredObjectCandidates(int absCharge = -2) const;
    /// returns only netural filtered objects
    std::vector<const Candidate*> neutralFilteredObjectCandidates() const;
    /// returns only charged filtered objects
    std::vector<const Candidate*> chargedFilteredObjectCandidates() const;

    /// fills master clones to PF objects (utility function)
    void pfMasterClones(const Candidate* input, PFCandidateRefVector& toFill) const;

    /// returns the PFCandidates associated to the charged signal objects
    PFCandidateRefVector associatedChargedPFCandidates() const;
    /// returns the PFCandidates associated to the PiZero signal objects (i.e., the unmerged photons)
    PFCandidateRefVector associatedNeutralPFCandidates() const;
    /// returns the PFCandidates that were filtered
    PFCandidateRefVector filteredPFCandidates() const;

  protected:
    PFTauRef pfTauRef_;
    VertexCompositeCandidate chargedPions_;
    CompositeCandidate piZeroes_;
    CompositeCandidate filteredObjects_;  // stores objects considered UE
    hadronicTauDecayModes theDecayMode_;
  };
}  // namespace reco

#endif
