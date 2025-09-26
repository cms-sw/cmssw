#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "DataFormats/L1Trigger/interface/P2GTAlgoBlock.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class HLTP2GTTauFilter : public HLTFilter {
public:
  explicit HLTP2GTTauFilter(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs&) const override;

private:
  edm::InputTag m_l1GTAlgoBlockTag;
  edm::EDGetTokenT<l1t::P2GTAlgoBlockMap> m_algoBlockToken;
  std::vector<std::string> m_l1GTAlgoNames;
  double m_minPt;
  unsigned int m_minN;
  double m_maxAbsEta;
};

HLTP2GTTauFilter::HLTP2GTTauFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  m_l1GTAlgoBlockTag = iConfig.getParameter<edm::InputTag>("l1GTAlgoBlockTag");
  m_algoBlockToken = consumes<l1t::P2GTAlgoBlockMap>(m_l1GTAlgoBlockTag);
  m_l1GTAlgoNames = iConfig.getParameter<std::vector<std::string>>("l1GTAlgoNames");
  m_minPt = iConfig.getParameter<double>("minPt");
  m_minN = iConfig.getParameter<unsigned int>("minN");
  m_maxAbsEta = iConfig.getParameter<double>("maxAbsEta");
}

void HLTP2GTTauFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("l1GTAlgoBlockTag", edm::InputTag(""));
  desc.add<std::vector<std::string>>("l1GTAlgoNames", {});
  desc.add<double>("minPt", 24);
  desc.add<unsigned int>("minN", 1);
  desc.add<double>("maxAbsEta", 1e99);
  descriptions.add("HLTP2GTTauFilter", desc);
}

bool HLTP2GTTauFilter::hltFilter(edm::Event& iEvent,
                                 const edm::EventSetup& iSetup,
                                 trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  std::vector<l1t::P2GTCandidateRef> vl1cands;
  if (saveTags()) {
    filterproduct.addCollectionTag(m_l1GTAlgoBlockTag);  // algorithm blocks
  }
  bool check_l1match = true;
  if (m_l1GTAlgoBlockTag.isUninitialized() || m_l1GTAlgoNames.empty())
    check_l1match = false;
  if (check_l1match) {
    const l1t::P2GTAlgoBlockMap& algos = iEvent.get(m_algoBlockToken);
    for (const auto& algoName : m_l1GTAlgoNames) {
      auto it = algos.find(algoName);
      if (it != algos.end() && it->second.decisionBeforeBxMaskAndPrescale()) {
        const l1t::P2GTCandidateVectorRef& objects = it->second.trigObjects();
        for (const l1t::P2GTCandidateRef& obj : objects) {
          if (obj->objectType() == l1t::P2GTCandidate::CL2Taus && obj->pt() >= m_minPt &&
              std::abs(obj->eta()) <= m_maxAbsEta) {
            vl1cands.push_back(obj);
          }
        }
      }
    }
  }

  if (saveTags()) {
    edm::InputTag tagOld;
    for (size_t i = 0; i < vl1cands.size(); ++i) {
      const auto& cand = vl1cands[i];
      const edm::ProductID pid(cand.id());
      const auto& prov = iEvent.getStableProvenance(pid);
      edm::InputTag tagNew(prov.moduleLabel(), prov.productInstanceName(), prov.processName());
      if (tagNew.encode() != tagOld.encode()) {
        filterproduct.addCollectionTag(tagNew);
        tagOld = tagNew;
      }
    }
  }
  for (const auto& vl1cand : vl1cands) {
    filterproduct.addObject(trigger::TriggerL1Tau, vl1cand);
  }

  return vl1cands.size() >= m_minN;
}
DEFINE_FWK_MODULE(HLTP2GTTauFilter);
