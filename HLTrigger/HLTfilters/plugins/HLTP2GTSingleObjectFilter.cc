// HLT filter seeded by a Phase-2 GT single-object condition.
// Mirrors l1tGTSingleObjectCond: one collection, per-object kinematic cuts.
//
// Example python config:
//
//   hltP2GTFilterCL2Taus = cms.EDFilter("HLTP2GTSingleObjectFilter",
//       l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
//       minN = cms.uint32(2),
//       l1GTAlgos = cms.VPSet(
//           cms.PSet(
//               name       = cms.string("pDoubleTau_Seed"),
//               collection = cms.PSet(
//                   objectType = cms.string("CL2Taus"),
//                   minPt      = cms.double(35.0),
//                   maxAbsEta  = cms.double(2.1),
//               ),
//           ),
//       ),
//   )

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

class HLTP2GTSingleObjectFilter : public HLTFilter {
public:
  explicit HLTP2GTSingleObjectFilter(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs&) const override;

private:
  struct AlgoConfig {
    std::string algoName;
    hltp2gt::CollectionSpec collection;
    AlgoConfig(const edm::ParameterSet& ps)
        : algoName(ps.getParameter<std::string>("name")),
          collection(ps.getParameter<edm::ParameterSet>("collection")) {}
  };

  const edm::InputTag m_algoBlockTag;
  const edm::EDGetTokenT<l1t::P2GTAlgoBlockMap> m_algoBlockToken;
  const unsigned int m_minN;
  const bool m_debugAccepts;
  std::vector<AlgoConfig> m_algos;
};

HLTP2GTSingleObjectFilter::HLTP2GTSingleObjectFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      m_algoBlockTag(iConfig.getParameter<edm::InputTag>("l1GTAlgoBlockTag")),
      m_algoBlockToken(consumes<l1t::P2GTAlgoBlockMap>(m_algoBlockTag)),
      m_minN(iConfig.getParameter<unsigned int>("minN")),
      m_debugAccepts(iConfig.getParameter<bool>("debugAccepts")) {
  for (const auto& ps : iConfig.getParameter<std::vector<edm::ParameterSet>>("l1GTAlgos"))
    m_algos.emplace_back(ps);
}

void HLTP2GTSingleObjectFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("l1GTAlgoBlockTag", edm::InputTag("l1tGTAlgoBlockProducer"));
  desc.add<unsigned int>("minN", 1);
  desc.add<bool>("debugAccepts", false);

  edm::ParameterSetDescription algoDesc;
  algoDesc.add<std::string>("name", "");
  edm::ParameterSetDescription collDesc;
  hltp2gt::CollectionSpec::fillDescription(collDesc, "CL2Taus");
  algoDesc.add<edm::ParameterSetDescription>("collection", collDesc);

  desc.addVPSet("l1GTAlgos", algoDesc, {});
  descriptions.addWithDefaultLabel(desc);
}

bool HLTP2GTSingleObjectFilter::hltFilter(edm::Event& iEvent,
                                          const edm::EventSetup&,
                                          trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  if (saveTags())
    filterproduct.addCollectionTag(m_algoBlockTag);

  if (m_algos.empty())
    return false;

  const auto& algoMap = iEvent.get(m_algoBlockToken);

  std::vector<l1t::P2GTCandidateRef> accepted;
  edm::InputTag lastTag;

  for (const auto& cfg : m_algos) {
    auto it = algoMap.find(cfg.algoName);
    if (it == algoMap.end()) {
      if (m_debugAccepts)
        edm::LogPrint("HLTP2GTSingleObjectFilter")
            << "[" << cfg.algoName << "] REJECT algo not found in P2GTAlgoBlockMap";
      continue;
    }
    if (!it->second.decisionBeforeBxMaskAndPrescale()) {
      if (m_debugAccepts)
        edm::LogPrint("HLTP2GTSingleObjectFilter")
            << "[" << cfg.algoName << "] REJECT decisionBeforeBxMaskAndPrescale=false"
            << " (trigObjects size=" << it->second.trigObjects().size() << ")";
      continue;
    }

    for (const auto& ref : it->second.trigObjects()) {
      if (m_debugAccepts) {
        cfg.collection.acceptsWithDebug(*ref, cfg.algoName);
        if (!cfg.collection.accepts(*ref))
          continue;
      } else if (!cfg.collection.accepts(*ref)) {
        continue;
      }
      if (saveTags())
        hltp2gt::addCollectionTagOnce(ref, iEvent, filterproduct, lastTag);
      accepted.push_back(ref);
    }
  }

  for (const auto& ref : accepted)
    filterproduct.addObject(hltp2gt::triggerTypeForP2GT(ref->objectType()), ref);

  LogDebug("HLTP2GTSingleObjectFilter") << "accepted " << accepted.size() << " objects (minN=" << m_minN << ")";

  return accepted.size() >= m_minN;
}

DEFINE_FWK_MODULE(HLTP2GTSingleObjectFilter);
