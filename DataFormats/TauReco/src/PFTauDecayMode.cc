#include "DataFormats/TauReco/interface/PFTauDecayMode.h"

namespace reco {
  PFTauDecayMode::PFTauDecayMode(const VertexCompositeCandidate& chargedPions,
                                 const CompositeCandidate& piZeroes,
                                 const CompositeCandidate& filteredObjects) {
    chargedPions_ = chargedPions;
    piZeroes_ = piZeroes;
    filteredObjects_ = filteredObjects;

    // determine decay mode
    unsigned int nCharged = chargedPions_.numberOfDaughters();
    unsigned int nNeutral = piZeroes_.numberOfDaughters();
    hadronicTauDecayModes hadronicTauDecayIndex =
        static_cast<hadronicTauDecayModes>(((nCharged - 1) * (maxNumberOfPiZeroCandidatesAllowed + 1) + nNeutral));
    if (nNeutral > maxNumberOfPiZeroCandidatesAllowed)
      hadronicTauDecayIndex = static_cast<hadronicTauDecayModes>(tauDecayOther);
    this->setDecayMode(hadronicTauDecayIndex);

    // setup Particle base
    for (size_type iCand = 0; iCand < nCharged; ++iCand) {
      const Candidate* theCandToAdd = chargedPions_.daughter(iCand);
      this->addDaughter(*theCandToAdd);
    }
    for (size_type iCand = 0; iCand < nNeutral; ++iCand) {
      const Candidate* theCandToAdd = piZeroes_.daughter(iCand);
      this->addDaughter(*theCandToAdd);
    }

    this->setCharge(chargedPions_.charge());
    this->setP4(chargedPions_.p4() + piZeroes_.p4());
    this->setStatus(2);  //decayed
    this->setPdgId(12);  //everyone's favorite lepton!
  }

  PFTauDecayMode* PFTauDecayMode::clone() const { return new PFTauDecayMode(*this); }

  const VertexCompositeCandidate& PFTauDecayMode::chargedPions() const { return chargedPions_; }

  const CompositeCandidate& PFTauDecayMode::neutralPions() const { return piZeroes_; }

  const CompositeCandidate& PFTauDecayMode::filteredObjects() const { return filteredObjects_; }

  std::vector<const Candidate*> PFTauDecayMode::chargedPionCandidates() const {
    size_type numberOfChargedPions = chargedPions_.numberOfDaughters();
    std::vector<const Candidate*> output;
    for (size_type iterCand = 0; iterCand < numberOfChargedPions; ++iterCand)
      output.push_back(chargedPions_.daughter(iterCand));
    return output;
  }

  std::vector<const Candidate*> PFTauDecayMode::neutralPionCandidates() const {
    size_type numberOfChargedPions = piZeroes_.numberOfDaughters();
    std::vector<const Candidate*> output;
    for (size_type iterCand = 0; iterCand < numberOfChargedPions; ++iterCand)
      output.push_back(piZeroes_.daughter(iterCand));
    return output;
  }

  std::vector<const Candidate*> PFTauDecayMode::decayProductCandidates() const {
    std::vector<const Candidate*> output = this->chargedPionCandidates();
    std::vector<const Candidate*> neutralObjects = this->neutralPionCandidates();

    output.insert(output.end(), neutralObjects.begin(), neutralObjects.end());
    return output;
  }

  std::vector<const Candidate*> PFTauDecayMode::filteredObjectCandidates(int absCharge) const {
    size_t numberOfFilteredObjects = filteredObjects_.numberOfDaughters();
    std::vector<const Candidate*> output;
    for (size_t iFilteredCand = 0; iFilteredCand < numberOfFilteredObjects; ++iFilteredCand) {
      const Candidate* myCand = filteredObjects_.daughter(iFilteredCand);
      if (absCharge < 0 || abs(myCand->charge()) == absCharge)
        output.push_back(myCand);
    }
    return output;
  }

  std::vector<const Candidate*> PFTauDecayMode::chargedFilteredObjectCandidates() const {
    return filteredObjectCandidates(1);
  }

  std::vector<const Candidate*> PFTauDecayMode::neutralFilteredObjectCandidates() const {
    return filteredObjectCandidates(0);
  }

  void PFTauDecayMode::pfMasterClones(const Candidate* input, PFCandidateRefVector& toFill) const {
    if (input->numberOfDaughters() == 0)  //we have reached a leaf
    {
      if (input->hasMasterClone())  // has a master clone
      {
        PFCandidateRef theCandRef = input->masterClone().castTo<PFCandidateRef>();
        toFill.push_back(theCandRef);
      } else
        edm::LogError("PFTauDecayMode")
            << "Error in pfMasterClones(...) - found a leaf candidate with no Master clone reference!";
    } else  // recurse down composite chain
    {
      size_type numberOfDaughters = input->numberOfDaughters();
      for (size_type iCand = 0; iCand < numberOfDaughters; ++iCand) {
        const Candidate* currentCand = input->daughter(iCand);
        pfMasterClones(currentCand, toFill);
      }
    }
  }

  PFCandidateRefVector PFTauDecayMode::associatedChargedPFCandidates() const {
    PFCandidateRefVector output;
    const Candidate* input = static_cast<const Candidate*>(&chargedPions_);
    if (input->numberOfDaughters())
      pfMasterClones(input, output);
    return output;
  }

  PFCandidateRefVector PFTauDecayMode::associatedNeutralPFCandidates() const {
    PFCandidateRefVector output;
    const Candidate* input = static_cast<const Candidate*>(&piZeroes_);
    if (input->numberOfDaughters())
      pfMasterClones(input, output);
    return output;
  }

  PFCandidateRefVector PFTauDecayMode::filteredPFCandidates() const {
    PFCandidateRefVector output;
    const Candidate* input = static_cast<const Candidate*>(&filteredObjects_);
    if (input->numberOfDaughters())
      pfMasterClones(input, output);
    return output;
  }

}  // namespace reco
