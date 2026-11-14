#ifndef HLTrigger_HLTcore_getL1TCandidates_h
#define HLTrigger_HLTcore_getL1TCandidates_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include <vector>

namespace hlt {

  // Returns the L1T objects stored with trigger type 'triggerType' in a
  // trigger::TriggerRefsCollections object (e.g. the trigger::TriggerFilterObjectWithRefs
  // product of an L1T seeding filter), as pointers to reco::Candidate.
  //
  // Two kinds of references are read:
  //  - references of type L1TVectorRef (e.g. l1t::JetVectorRef or l1t::TauVectorRef for the
  //    Run-3 L1T objects stored by HLTL1TSeed);
  //  - references to Phase-2 L1T objects (l1t::P2GTCandidateRef), stored by
  //    HLTP2GTSingleObjectFilter and the other HLTP2GT*Filter modules in a separate container,
  //    with the trigger type given by HLTP2GTUtilities.h (e.g. CL2JetsSC4 -> TriggerL1Jet,
  //    CL2Taus -> TriggerL1Tau, GCTTaus and GTTHadronicTaus -> TriggerL1TauJet,
  //    CL2Electrons and CL2Photons -> TriggerL1EG).
  // This lets HLT modules that match HLT objects to the seeding L1T objects work for both Run-3
  // and Phase-2 seeds. The objects are returned as reco::Candidate pointers (both kinds are
  // reco::LeafCandidate), so that callers keep using the same accessors (p4(), polarP4(), pt(),
  // eta(), phi()) and get exactly the stored values. The pointers are valid as long as the
  // event products they point to. The L1TVectorRef objects come first in the output.
  template <typename L1TVectorRef>
  std::vector<reco::Candidate const*> getL1TCandidates(trigger::TriggerRefsCollections const& triggerRefs,
                                                       int const triggerType) {
    L1TVectorRef l1tRefs;
    triggerRefs.getObjects(triggerType, l1tRefs);

    l1t::P2GTCandidateVectorRef l1tP2GTCandRefs;
    triggerRefs.getObjects(triggerType, l1tP2GTCandRefs);

    std::vector<reco::Candidate const*> ret;
    ret.reserve(l1tRefs.size() + l1tP2GTCandRefs.size());

    for (auto const& ref : l1tRefs)
      ret.emplace_back(ref.get());

    for (auto const& ref : l1tP2GTCandRefs)
      ret.emplace_back(ref.get());

    return ret;
  }

  // L1T e/gammas: l1t::EGamma (Run-3) and l1t::P2GTCandidate (Phase-2).
  // Note: for Phase-2 seeds, TriggerL1EG covers CL2Electrons and CL2Photons
  // (GCT e/gammas are TriggerL1IsoEG and TriggerL1NoIsoEG).
  inline std::vector<reco::Candidate const*> getL1TEGammas(trigger::TriggerRefsCollections const& triggerRefs,
                                                           int const triggerType = trigger::TriggerL1EG) {
    return getL1TCandidates<l1t::EGammaVectorRef>(triggerRefs, triggerType);
  }

  // L1T jets: l1t::Jet (Run-3) and l1t::P2GTCandidate (Phase-2).
  // Note: for Phase-2 seeds, TriggerL1Jet also covers the GTT phi, rho and Bs candidates.
  inline std::vector<reco::Candidate const*> getL1TJets(trigger::TriggerRefsCollections const& triggerRefs,
                                                        int const triggerType = trigger::TriggerL1Jet) {
    return getL1TCandidates<l1t::JetVectorRef>(triggerRefs, triggerType);
  }

  // L1T taus: l1t::Tau (Run-3) and l1t::P2GTCandidate (Phase-2).
  // Note: for Phase-2 seeds, TriggerL1Tau covers CL2Taus only (GCT and GTT taus are TriggerL1TauJet).
  inline std::vector<reco::Candidate const*> getL1TTaus(trigger::TriggerRefsCollections const& triggerRefs,
                                                        int const triggerType = trigger::TriggerL1Tau) {
    return getL1TCandidates<l1t::TauVectorRef>(triggerRefs, triggerType);
  }

}  // namespace hlt

#endif  // HLTrigger_HLTcore_getL1TCandidates_h
