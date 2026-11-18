#ifndef HLTrigger_HLTcore_getL1TCandidates_h
#define HLTrigger_HLTcore_getL1TCandidates_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include <type_traits>
#include <vector>

namespace hlt {

  // Returns the L1T objects stored with trigger type 'triggerType' in a
  // trigger::TriggerRefsCollections object (e.g. the trigger::TriggerFilterObjectWithRefs
  // product of an L1T seeding filter), as pointers to reco::Candidate.
  //
  // Two kinds of references are read:
  //  - references of the types L1TVectorRefs, if any (e.g. l1t::JetVectorRef or
  //    l1t::TauVectorRef for the Run-3 L1T objects stored by HLTL1TSeed);
  //  - references to Phase-2 L1T objects (l1t::P2GTCandidateRef), always. They are stored by
  //    HLTP2GTSingleObjectFilter and the other HLTP2GT*Filter modules in a separate container,
  //    with the trigger type given by HLTP2GTUtilities.h (e.g. CL2JetsSC4 -> TriggerL1Jet,
  //    CL2Taus -> TriggerL1Tau, GCTTaus and GTTHadronicTaus -> TriggerL1TauJet,
  //    CL2Electrons and CL2Photons -> TriggerL1EG).
  // For example, getL1TCandidates<>(triggerRefs, trigger::TriggerL1TauJet) reads only the
  // Phase-2 GCT and GTT taus.
  // This lets HLT modules that match HLT objects to the seeding L1T objects work for both Run-3
  // and Phase-2 seeds. The objects are returned as reco::Candidate pointers (both kinds are
  // reco::LeafCandidate), so that callers keep using the same accessors (p4(), polarP4(), pt(),
  // eta(), phi()) and get exactly the stored values. The pointers are valid as long as the
  // event products they point to. The objects come in the order of L1TVectorRefs, followed by
  // the Phase-2 ones.
  template <typename... L1TVectorRefs>
  std::vector<reco::Candidate const*> getL1TCandidates(trigger::TriggerRefsCollections const& triggerRefs,
                                                       int const triggerType) {
    std::vector<reco::Candidate const*> ret;
    // appends the objects of the ref type given by the tag (std::type_identity<VectorRef>)
    auto append = [&](auto tag) {
      typename decltype(tag)::type refs;
      triggerRefs.getObjects(triggerType, refs);
      ret.reserve(ret.size() + refs.size());
      for (auto const& ref : refs)
        ret.emplace_back(ref.get());
    };
    (append(std::type_identity<L1TVectorRefs>{}), ...);
    append(std::type_identity<l1t::P2GTCandidateVectorRef>{});
    return ret;
  }

  // L1T e/gammas: l1t::EGamma (Run-3, TriggerL1EG) and l1t::P2GTCandidate (Phase-2).
  // For Phase-2 seeds, TriggerL1EG covers CL2Electrons and CL2Photons, while the GCT e/gammas
  // are stored as TriggerL1IsoEG and TriggerL1NoIsoEG: the l1t::P2GTCandidate refs of these two
  // types are read as well (Run-3 seeds store no l1t::EGamma refs with these types).
  inline std::vector<reco::Candidate const*> getL1TEGammas(trigger::TriggerRefsCollections const& triggerRefs) {
    auto ret = getL1TCandidates<l1t::EGammaVectorRef>(triggerRefs, trigger::TriggerL1EG);
    for (int const triggerType : {trigger::TriggerL1IsoEG, trigger::TriggerL1NoIsoEG}) {
      auto const gctEGammas = getL1TCandidates<>(triggerRefs, triggerType);
      ret.insert(ret.end(), gctEGammas.begin(), gctEGammas.end());
    }
    return ret;
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
