// HLT filter seeded by a Phase-2 GT triple-object condition.
// Mirrors l1tGTTripleObjectCond: three collections (may all be different types),
// per-object kinematic cuts, and three independent pair-cut PSets:
//   cuts12: between collection1 and collection2
//   cuts13: between collection1 and collection3
//   cuts23: between collection2 and collection3
//
// Example python config:
//
//   hltP2GTFilterTripleMu = cms.EDFilter("HLTP2GTTripleObjectFilter",
//       l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
//       l1GTAlgos = cms.VPSet(
//           cms.PSet(
//               name = cms.string("pTripleMu_5_3_3"),
//               collection1 = cms.PSet(objectType=cms.string("GMTTkMuons"),
//                                     minPt=cms.double(5.), maxAbsEta=cms.double(2.4)),
//               collection2 = cms.PSet(objectType=cms.string("GMTTkMuons"),
//                                     minPt=cms.double(3.), maxAbsEta=cms.double(2.4)),
//               collection3 = cms.PSet(objectType=cms.string("GMTTkMuons"),
//                                     minPt=cms.double(3.), maxAbsEta=cms.double(2.4)),
//               cuts12 = cms.PSet(minDR=cms.double(0.), maxDR=cms.double(1e9),
//                                 minDEta=cms.double(-1.), minDPhi=cms.double(-1.),
//                                 minInvMass=cms.double(0.), maxInvMass=cms.double(1e9)),
//               cuts13 = cms.PSet(minDR=cms.double(0.), maxDR=cms.double(1e9),
//                                 minDEta=cms.double(-1.), minDPhi=cms.double(-1.),
//                                 minInvMass=cms.double(0.), maxInvMass=cms.double(1e9)),
//               cuts23 = cms.PSet(minDR=cms.double(0.), maxDR=cms.double(1e9),
//                                 minDEta=cms.double(-1.), minDPhi=cms.double(-1.),
//                                 minInvMass=cms.double(0.), maxInvMass=cms.double(1e9)),
//           ),
//       ),
//   )
//
// Semantics:
//   For each firing algo iterate all ordered triples (o1,o2,o3) where
//   o1 in coll1, o2 in coll2, o3 in coll3 and all three are distinct.
//   The filter passes if at least one valid triple is found across all algos.

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

class HLTP2GTTripleObjectFilter : public HLTFilter {
public:
  explicit HLTP2GTTripleObjectFilter(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  bool hltFilter(edm::Event&, const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs&) const override;

private:
  struct AlgoConfig {
    std::string algoName;
    hltp2gt::CollectionSpec coll1;
    hltp2gt::CollectionSpec coll2;
    hltp2gt::CollectionSpec coll3;
    hltp2gt::PairCuts cuts12;
    hltp2gt::PairCuts cuts13;
    hltp2gt::PairCuts cuts23;

    AlgoConfig(const edm::ParameterSet& ps)
        : algoName(ps.getParameter<std::string>("name")),
          coll1(ps.getParameter<edm::ParameterSet>("collection1")),
          coll2(ps.getParameter<edm::ParameterSet>("collection2")),
          coll3(ps.getParameter<edm::ParameterSet>("collection3")),
          cuts12(ps.getParameter<edm::ParameterSet>("cuts12")),
          cuts13(ps.getParameter<edm::ParameterSet>("cuts13")),
          cuts23(ps.getParameter<edm::ParameterSet>("cuts23")) {}
  };

  const edm::InputTag m_algoBlockTag;
  const edm::EDGetTokenT<l1t::P2GTAlgoBlockMap> m_algoBlockToken;
  std::vector<AlgoConfig> m_algos;
};

HLTP2GTTripleObjectFilter::HLTP2GTTripleObjectFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      m_algoBlockTag(iConfig.getParameter<edm::InputTag>("l1GTAlgoBlockTag")),
      m_algoBlockToken(consumes<l1t::P2GTAlgoBlockMap>(m_algoBlockTag)) {
  for (const auto& ps : iConfig.getParameter<std::vector<edm::ParameterSet>>("l1GTAlgos"))
    m_algos.emplace_back(ps);
}

void HLTP2GTTripleObjectFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("l1GTAlgoBlockTag", edm::InputTag("l1tGTAlgoBlockProducer"));

  edm::ParameterSetDescription algoDesc;
  algoDesc.add<std::string>("name", "");

  // Three collection sub-PSets
  edm::ParameterSetDescription cDesc;
  hltp2gt::CollectionSpec::fillDescription(cDesc, "GMTTkMuons");
  algoDesc.add<edm::ParameterSetDescription>("collection1", cDesc);
  algoDesc.add<edm::ParameterSetDescription>("collection2", cDesc);
  algoDesc.add<edm::ParameterSetDescription>("collection3", cDesc);

  // Three pair-cut sub-PSets
  edm::ParameterSetDescription pcDesc;
  hltp2gt::PairCuts::fillDescription(pcDesc);
  algoDesc.add<edm::ParameterSetDescription>("cuts12", pcDesc);
  algoDesc.add<edm::ParameterSetDescription>("cuts13", pcDesc);
  algoDesc.add<edm::ParameterSetDescription>("cuts23", pcDesc);

  desc.addVPSet("l1GTAlgos", algoDesc, {});
  descriptions.addWithDefaultLabel(desc);
}

bool HLTP2GTTripleObjectFilter::hltFilter(edm::Event& iEvent,
                                          const edm::EventSetup&,
                                          trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  if (saveTags())
    filterproduct.addCollectionTag(m_algoBlockTag);

  if (m_algos.empty())
    return false;

  const auto& algoMap = iEvent.get(m_algoBlockToken);

  std::vector<l1t::P2GTCandidateRef> matched1, matched2, matched3;
  edm::InputTag lastTag;

  for (const auto& cfg : m_algos) {
    auto it = algoMap.find(cfg.algoName);
    if (it == algoMap.end() || !it->second.decisionBeforeBxMaskAndPrescale())
      continue;

    const auto& objs = it->second.trigObjects();

    for (std::size_t i = 0; i < objs.size(); ++i) {
      const auto& r1 = objs[i];
      if (!cfg.coll1.accepts(*r1)) continue;
      for (std::size_t j = 0; j < objs.size(); ++j) {
        if (j == i) continue;
        const auto& r2 = objs[j];
        // Suppress the (j,i) duplicate only when r1 and r2 come from the
        // same underlying product.  Different products (e.g. barrel vs
        // forward jets with the same ObjectType) must both be tested.
        if (r1.id() == r2.id() && j < i) continue;
        if (!cfg.coll2.accepts(*r2)) continue;
        if (!cfg.cuts12.accepts(*r1, *r2)) continue;
        for (std::size_t k = 0; k < objs.size(); ++k) {
          if (k == i || k == j) continue;
          const auto& r3 = objs[k];
          // Apply the same per-pair ProductID deduplication for (1,3) and (2,3).
          if (r1.id() == r3.id() && k < i) continue;
          if (r2.id() == r3.id() && k < j) continue;
          if (!cfg.coll3.accepts(*r3)) continue;
          if (!cfg.cuts13.accepts(*r1, *r3)) continue;
          if (!cfg.cuts23.accepts(*r2, *r3)) continue;

          LogDebug("HLTP2GTTripleObjectFilter")
              << "  accepted triple: "
              << hltp2gt::objectTypeName(r1->objectType()) << " pT=" << r1->pt()
              << "  x  "
              << hltp2gt::objectTypeName(r2->objectType()) << " pT=" << r2->pt()
              << "  x  "
              << hltp2gt::objectTypeName(r3->objectType()) << " pT=" << r3->pt();

          if (saveTags()) {
            hltp2gt::addCollectionTagOnce(r1, iEvent, filterproduct, lastTag);
            hltp2gt::addCollectionTagOnce(r2, iEvent, filterproduct, lastTag);
            hltp2gt::addCollectionTagOnce(r3, iEvent, filterproduct, lastTag);
          }
          matched1.push_back(r1);
          matched2.push_back(r2);
          matched3.push_back(r3);
        }  // k
      }  // j
    }  // i
  }

  for (const auto& ref : matched1)
    filterproduct.addObject(hltp2gt::triggerTypeForP2GT(ref->objectType()), ref);
  for (const auto& ref : matched2)
    filterproduct.addObject(hltp2gt::triggerTypeForP2GT(ref->objectType()), ref);
  for (const auto& ref : matched3)
    filterproduct.addObject(hltp2gt::triggerTypeForP2GT(ref->objectType()), ref);

  const bool pass = !matched1.empty();
  LogDebug("HLTP2GTTripleObjectFilter")
      << "found " << matched1.size() << " triples, result=" << pass;
  return pass;
}

DEFINE_FWK_MODULE(HLTP2GTTripleObjectFilter);
