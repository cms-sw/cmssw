
#include "HLTJetPairDzMatchFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template <typename T>
HLTJetPairDzMatchFilter<T>::HLTJetPairDzMatchFilter(const edm::ParameterSet& conf) : HLTFilter(conf) {
  m_jetTag = conf.getParameter<edm::InputTag>("JetSrc");
  m_jetToken = consumes<std::vector<T>>(m_jetTag);
  m_jetMinPt = conf.getParameter<double>("JetMinPt");
  m_jetMaxEta = conf.getParameter<double>("JetMaxEta");
  m_jetMinDR = conf.getParameter<double>("JetMinDR");
  m_jetMaxDZ = conf.getParameter<double>("JetMaxDZ");
  m_triggerType = conf.getParameter<int>("TriggerType");

  // set the minimum DR between jets, so that one never has a chance
  // to create a pair out of the same Jet replicated with two
  // different vertices
  if (m_jetMinDR < 0.1)
    m_jetMinDR = 0.1;
}

template <typename T>
void HLTJetPairDzMatchFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("JetSrc", edm::InputTag("hltMatchL2Tau30ToPixelTrk5"));
  desc.add<double>("JetMinPt", 25.0);
  desc.add<double>("JetMaxEta", 2.4);
  desc.add<double>("JetMinDR", 0.2);
  desc.add<double>("JetMaxDZ", 0.2);
  desc.add<int>("TriggerType", 84);
  descriptions.add(defaultModuleLabel<HLTJetPairDzMatchFilter<T>>(), desc);
}

template <typename T>
HLTJetPairDzMatchFilter<T>::~HLTJetPairDzMatchFilter() = default;

template <typename T>
bool HLTJetPairDzMatchFilter<T>::hltFilter(edm::Event& ev,
                                           const edm::EventSetup& es,
                                           trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;

  // The resuilting filter object to store in the Event

  if (saveTags())
    filterproduct.addCollectionTag(m_jetTag);

  // Ref to Candidate object to be recorded in the filter object
  TRef ref;

  // *** Pick up L2 tau jets which have been equipped with some meaningful vertices before that ***

  edm::Handle<TCollection> jetsHandle;
  ev.getByToken(m_jetToken, jetsHandle);
  const TCollection& jets = *jetsHandle;
  const size_t n_jets = jets.size();

  // *** Combine jets into pairs and check the dz matching ***

  size_t npairs = 0;
  if (n_jets > 1)
    for (size_t j1 = 0; j1 < n_jets; ++j1) {
      if (jets[j1].pt() < m_jetMinPt || std::abs(jets[j1].eta()) > m_jetMaxEta)
        continue;

      float mindz = 99.f;
      for (size_t j2 = j1 + 1; j2 < n_jets; ++j2) {
        if (jets[j2].pt() < m_jetMinPt || std::abs(jets[j2].eta()) > m_jetMaxEta)
          continue;

        float deta = jets[j1].eta() - jets[j2].eta();
        float dphi = reco::deltaPhi(jets[j1].phi(), jets[j2].phi());
        float dr2 = dphi * dphi + deta * deta;
        float dz = jets[j1].vz() - jets[j2].vz();

        // skip pairs of jets that are close
        if (dr2 < m_jetMinDR * m_jetMinDR) {
          continue;
        }

        if (std::abs(dz) < std::abs(mindz))
          mindz = dz;

        // do not form a pair if dz is too large
        if (std::abs(dz) > m_jetMaxDZ) {
          continue;
        }

        // add references to both jets
        ref = TRef(jetsHandle, j1);
        filterproduct.addObject(m_triggerType, ref);

        ref = TRef(jetsHandle, j2);
        filterproduct.addObject(m_triggerType, ref);

        ++npairs;
      }
    }

  return npairs > 0;
}
