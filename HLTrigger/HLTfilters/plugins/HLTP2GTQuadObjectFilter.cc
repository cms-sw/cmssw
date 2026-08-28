// HLT filter seeded by a Phase-2 GT quad-object condition.
// Mirrors l1tGTQuadObjectCond: four collections (may all be different types),
// per-object kinematic cuts, and six independent pair-cut PSets covering
// all combinations:
//   cuts12, cuts13, cuts14, cuts23, cuts24, cuts34
//
// Example python config:
//
//   hltP2GTFilterQuadJet70554040 = cms.EDFilter("HLTP2GTQuadObjectFilter",
//       l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
//       l1GTAlgos = cms.VPSet(
//           cms.PSet(
//               name = cms.string("pQuadJet70_55_40_40"),
//               collection1 = cms.PSet(objectType=cms.string("CL2JetsSC4"),
//                                      minPt=cms.double(70.), maxAbsEta=cms.double(2.4)),
//               collection2 = cms.PSet(objectType=cms.string("CL2JetsSC4"),
//                                      minPt=cms.double(55.), maxAbsEta=cms.double(2.4)),
//               collection3 = cms.PSet(objectType=cms.string("CL2JetsSC4"),
//                                      minPt=cms.double(40.), maxAbsEta=cms.double(2.4)),
//               collection4 = cms.PSet(objectType=cms.string("CL2JetsSC4"),
//                                      minPt=cms.double(40.), maxAbsEta=cms.double(2.4)),
//               cuts12 = cms.PSet(minDR=cms.double(0.), maxDR=cms.double(1e9),
//                                 minDEta=cms.double(-1.), minDPhi=cms.double(-1.),
//                                 minInvMass=cms.double(0.), maxInvMass=cms.double(1e9)),
//               cuts13 = cms.PSet(...),
//               cuts14 = cms.PSet(...),
//               cuts23 = cms.PSet(...),
//               cuts24 = cms.PSet(...),
//               cuts34 = cms.PSet(...),
//           ),
//       ),
//   )
//
// Semantics:
//   For each firing algo iterate all 4-tuples (o1, o2, o3, o4) where
//   oi in colli and all four indices are distinct.
//   Deduplication: when two roles draw from the same underlying product
//   (same edm::ProductID) only the index-ordered assignment is kept,
//   exactly as in HLTP2GTDoubleObjectFilter and HLTP2GTTripleObjectFilter.
//   The filter passes if at least one valid 4-tuple is found.

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

class HLTP2GTQuadObjectFilter : public HLTFilter {
public:
  explicit HLTP2GTQuadObjectFilter(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  bool hltFilter(edm::Event&, const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs&) const override;

private:
  struct AlgoConfig {
    std::string algoName;
    hltp2gt::CollectionSpec coll1;
    hltp2gt::CollectionSpec coll2;
    hltp2gt::CollectionSpec coll3;
    hltp2gt::CollectionSpec coll4;
    hltp2gt::PairCuts cuts12;
    hltp2gt::PairCuts cuts13;
    hltp2gt::PairCuts cuts14;
    hltp2gt::PairCuts cuts23;
    hltp2gt::PairCuts cuts24;
    hltp2gt::PairCuts cuts34;

    AlgoConfig(const edm::ParameterSet& ps)
        : algoName(ps.getParameter<std::string>("name")),
          coll1(ps.getParameter<edm::ParameterSet>("collection1")),
          coll2(ps.getParameter<edm::ParameterSet>("collection2")),
          coll3(ps.getParameter<edm::ParameterSet>("collection3")),
          coll4(ps.getParameter<edm::ParameterSet>("collection4")),
          cuts12(ps.getParameter<edm::ParameterSet>("cuts12")),
          cuts13(ps.getParameter<edm::ParameterSet>("cuts13")),
          cuts14(ps.getParameter<edm::ParameterSet>("cuts14")),
          cuts23(ps.getParameter<edm::ParameterSet>("cuts23")),
          cuts24(ps.getParameter<edm::ParameterSet>("cuts24")),
          cuts34(ps.getParameter<edm::ParameterSet>("cuts34")) {}
  };

  const edm::InputTag m_algoBlockTag;
  const edm::EDGetTokenT<l1t::P2GTAlgoBlockMap> m_algoBlockToken;
  std::vector<AlgoConfig> m_algos;
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
HLTP2GTQuadObjectFilter::HLTP2GTQuadObjectFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      m_algoBlockTag(iConfig.getParameter<edm::InputTag>("l1GTAlgoBlockTag")),
      m_algoBlockToken(consumes<l1t::P2GTAlgoBlockMap>(m_algoBlockTag)) {
  for (const auto& ps : iConfig.getParameter<std::vector<edm::ParameterSet>>("l1GTAlgos"))
    m_algos.emplace_back(ps);
}

// ---------------------------------------------------------------------------
// fillDescriptions
// ---------------------------------------------------------------------------
void HLTP2GTQuadObjectFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("l1GTAlgoBlockTag", edm::InputTag("l1tGTAlgoBlockProducer"));

  edm::ParameterSetDescription algoDesc;
  algoDesc.add<std::string>("name", "");

  edm::ParameterSetDescription cDesc;
  hltp2gt::CollectionSpec::fillDescription(cDesc, "CL2JetsSC4");
  algoDesc.add<edm::ParameterSetDescription>("collection1", cDesc);
  algoDesc.add<edm::ParameterSetDescription>("collection2", cDesc);
  algoDesc.add<edm::ParameterSetDescription>("collection3", cDesc);
  algoDesc.add<edm::ParameterSetDescription>("collection4", cDesc);

  edm::ParameterSetDescription pcDesc;
  hltp2gt::PairCuts::fillDescription(pcDesc);
  algoDesc.add<edm::ParameterSetDescription>("cuts12", pcDesc);
  algoDesc.add<edm::ParameterSetDescription>("cuts13", pcDesc);
  algoDesc.add<edm::ParameterSetDescription>("cuts14", pcDesc);
  algoDesc.add<edm::ParameterSetDescription>("cuts23", pcDesc);
  algoDesc.add<edm::ParameterSetDescription>("cuts24", pcDesc);
  algoDesc.add<edm::ParameterSetDescription>("cuts34", pcDesc);

  desc.addVPSet("l1GTAlgos", algoDesc, {});
  descriptions.addWithDefaultLabel(desc);
}

// ---------------------------------------------------------------------------
// hltFilter
// ---------------------------------------------------------------------------
bool HLTP2GTQuadObjectFilter::hltFilter(edm::Event& iEvent,
                                        const edm::EventSetup&,
                                        trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  if (saveTags())
    filterproduct.addCollectionTag(m_algoBlockTag);

  if (m_algos.empty())
    return false;

  const auto& algoMap = iEvent.get(m_algoBlockToken);

  std::vector<l1t::P2GTCandidateRef> matched1, matched2, matched3, matched4;
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
        // Dedup (1,2): same product -> keep i < j only
        if (r1.id() == objs[j].id() && j < i) continue;
        const auto& r2 = objs[j];
        if (!cfg.coll2.accepts(*r2)) continue;
        if (!cfg.cuts12.accepts(*r1, *r2)) continue;

        for (std::size_t k = 0; k < objs.size(); ++k) {
          if (k == i || k == j) continue;
          // Dedup (1,3) and (2,3) independently
          if (r1.id() == objs[k].id() && k < i) continue;
          if (r2.id() == objs[k].id() && k < j) continue;
          const auto& r3 = objs[k];
          if (!cfg.coll3.accepts(*r3)) continue;
          if (!cfg.cuts13.accepts(*r1, *r3)) continue;
          if (!cfg.cuts23.accepts(*r2, *r3)) continue;

          for (std::size_t l = 0; l < objs.size(); ++l) {
            if (l == i || l == j || l == k) continue;
            // Dedup (1,4), (2,4), (3,4) independently
            if (r1.id() == objs[l].id() && l < i) continue;
            if (r2.id() == objs[l].id() && l < j) continue;
            if (r3.id() == objs[l].id() && l < k) continue;
            const auto& r4 = objs[l];
            if (!cfg.coll4.accepts(*r4)) continue;
            if (!cfg.cuts14.accepts(*r1, *r4)) continue;
            if (!cfg.cuts24.accepts(*r2, *r4)) continue;
            if (!cfg.cuts34.accepts(*r3, *r4)) continue;

            LogDebug("HLTP2GTQuadObjectFilter")
                << "  accepted quad: "
                << hltp2gt::objectTypeName(r1->objectType()) << " pT=" << r1->pt()
                << "  x  " << hltp2gt::objectTypeName(r2->objectType()) << " pT=" << r2->pt()
                << "  x  " << hltp2gt::objectTypeName(r3->objectType()) << " pT=" << r3->pt()
                << "  x  " << hltp2gt::objectTypeName(r4->objectType()) << " pT=" << r4->pt();

            if (saveTags()) {
              hltp2gt::addCollectionTagOnce(r1, iEvent, filterproduct, lastTag);
              hltp2gt::addCollectionTagOnce(r2, iEvent, filterproduct, lastTag);
              hltp2gt::addCollectionTagOnce(r3, iEvent, filterproduct, lastTag);
              hltp2gt::addCollectionTagOnce(r4, iEvent, filterproduct, lastTag);
            }
            matched1.push_back(r1);
            matched2.push_back(r2);
            matched3.push_back(r3);
            matched4.push_back(r4);
          }  // l
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
  for (const auto& ref : matched4)
    filterproduct.addObject(hltp2gt::triggerTypeForP2GT(ref->objectType()), ref);

  const bool pass = !matched1.empty();
  LogDebug("HLTP2GTQuadObjectFilter")
      << "found " << matched1.size() << " quads, result=" << pass;
  return pass;
}

DEFINE_FWK_MODULE(HLTP2GTQuadObjectFilter);
