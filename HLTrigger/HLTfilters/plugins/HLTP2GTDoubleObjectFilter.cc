// HLT filter seeded by a Phase-2 GT double-object condition.
// Mirrors l1tGTDoubleObjectCond: two collections (may be different types),
// per-object kinematic cuts, and inter-object cuts (DeltaR, DeltaEta, DeltaPhi, m_inv).
//
// Example python config:
//
//   hltP2GTFilterIsoTkEleEGEle2212 = cms.EDFilter("HLTP2GTDoubleObjectFilter",
//       l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
//       l1GTAlgos = cms.VPSet(
//           cms.PSet(
//               name = cms.string("pIsoTkEleEGEle22_12"),
//               collection1 = cms.PSet(
//                   objectType = cms.string("CL2Electrons"),
//                   minPt      = cms.double(22.0),
//                   maxAbsEta  = cms.double(2.4),
//               ),
//               collection2 = cms.PSet(
//                   objectType = cms.string("CL2Photons"),
//                   minPt      = cms.double(12.0),
//                   maxAbsEta  = cms.double(2.4),
//               ),
//               minDR      = cms.double(0.1),
//               maxDR      = cms.double(1e9),
//               minDEta    = cms.double(-1.0),
//               minDPhi    = cms.double(-1.0),
//               minInvMass = cms.double(0.0),
//               maxInvMass = cms.double(1e9),
//           ),
//       ),
//   )
//
// Semantics:
//   For each firing algo, iterate all ordered pairs (o1 in coll1, o2 in coll2).
//   When collection1 and collection2 have the same objectType the pair (o1,o2)
//   and (o2,o1) are both tested but a candidate can be accepted more than once —
//   this matches the upstream L1 behaviour.  Set minDR > 0 to suppress
//   self-pairs if both collections draw from the same physical list.
//
//   The filter passes if at least one valid pair is found across all algos.

#include "HLTP2GTUtilities.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "DataFormats/L1Trigger/interface/P2GTAlgoBlock.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <string>
#include <vector>

class HLTP2GTDoubleObjectFilter : public HLTFilter {
public:
  explicit HLTP2GTDoubleObjectFilter(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs&) const override;

private:
  struct AlgoConfig {
    std::string algoName;
    hltp2gt::CollectionSpec coll1;
    hltp2gt::CollectionSpec coll2;
    hltp2gt::PairCuts pairCuts;
    AlgoConfig(const edm::ParameterSet& ps)
        : algoName(ps.getParameter<std::string>("name")),
          coll1(ps.getParameter<edm::ParameterSet>("collection1")),
          coll2(ps.getParameter<edm::ParameterSet>("collection2")),
          pairCuts(ps) {}
  };

  const edm::InputTag m_algoBlockTag;
  const edm::EDGetTokenT<l1t::P2GTAlgoBlockMap> m_algoBlockToken;
  std::vector<AlgoConfig> m_algos;
};

HLTP2GTDoubleObjectFilter::HLTP2GTDoubleObjectFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      m_algoBlockTag(iConfig.getParameter<edm::InputTag>("l1GTAlgoBlockTag")),
      m_algoBlockToken(consumes<l1t::P2GTAlgoBlockMap>(m_algoBlockTag)) {
  for (const auto& ps : iConfig.getParameter<std::vector<edm::ParameterSet>>("l1GTAlgos"))
    m_algos.emplace_back(ps);
}

void HLTP2GTDoubleObjectFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("l1GTAlgoBlockTag", edm::InputTag("l1tGTAlgoBlockProducer"));

  edm::ParameterSetDescription algoDesc;
  algoDesc.add<std::string>("name", "");

  edm::ParameterSetDescription coll1Desc, coll2Desc;
  hltp2gt::CollectionSpec::fillDescription(coll1Desc, "CL2Electrons");
  hltp2gt::CollectionSpec::fillDescription(coll2Desc, "CL2Photons");
  algoDesc.add<edm::ParameterSetDescription>("collection1", coll1Desc);
  algoDesc.add<edm::ParameterSetDescription>("collection2", coll2Desc);

  hltp2gt::PairCuts::fillDescription(algoDesc);

  desc.addVPSet("l1GTAlgos", algoDesc, {});
  descriptions.addWithDefaultLabel(desc);
}

bool HLTP2GTDoubleObjectFilter::hltFilter(edm::Event& iEvent,
                                          const edm::EventSetup&,
                                          trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  if (saveTags())
    filterproduct.addCollectionTag(m_algoBlockTag);

  if (m_algos.empty())
    return false;

  const auto& algoMap = iEvent.get(m_algoBlockToken);

  // Collect matched refs separately per role so they carry the right type.
  std::vector<l1t::P2GTCandidateRef> matched1, matched2;
  edm::InputTag lastTag;

  for (const auto& cfg : m_algos) {
    auto it = algoMap.find(cfg.algoName);
    if (it == algoMap.end() || !it->second.decisionBeforeBxMaskAndPrescale())
      continue;

    const auto& objs = it->second.trigObjects();

    for (std::size_t i = 0; i < objs.size(); ++i) {
      const auto& r1 = objs[i];
      if (!cfg.coll1.accepts(*r1))
        continue;
      for (std::size_t j = 0; j < objs.size(); ++j) {
        if (j == i)
          continue;  // always skip self
        const auto& r2 = objs[j];
        // When r1 and r2 come from the same underlying product they are
        // interchangeable as coll1/coll2 candidates, so (i,j) and (j,i)
        // would be duplicates.  Enforce i < j in that case only.
        // For refs from different products both orderings are distinct
        // (e.g. barrel jet + forward jet) and must both be tested.
        if (r1.id() == r2.id() && j < i)
          continue;
        if (!cfg.coll2.accepts(*r2))
          continue;
        if (!cfg.pairCuts.accepts(*r1, *r2))
          continue;

        LogDebug("HLTP2GTDoubleObjectFilter")
            << "  accepted pair: " << hltp2gt::objectTypeName(r1->objectType()) << " pT=" << r1->pt() << "  x  "
            << hltp2gt::objectTypeName(r2->objectType()) << " pT=" << r2->pt();

        if (saveTags()) {
          hltp2gt::addCollectionTagOnce(r1, iEvent, filterproduct, lastTag);
          hltp2gt::addCollectionTagOnce(r2, iEvent, filterproduct, lastTag);
        }
        matched1.push_back(r1);
        matched2.push_back(r2);
      }
    }
  }

  for (const auto& ref : matched1)
    filterproduct.addObject(hltp2gt::triggerTypeForP2GT(ref->objectType()), ref);
  for (const auto& ref : matched2)
    filterproduct.addObject(hltp2gt::triggerTypeForP2GT(ref->objectType()), ref);

  const bool pass = !matched1.empty();
  LogDebug("HLTP2GTDoubleObjectFilter") << "found " << matched1.size() << " pairs, result=" << pass;
  return pass;
}

DEFINE_FWK_MODULE(HLTP2GTDoubleObjectFilter);
