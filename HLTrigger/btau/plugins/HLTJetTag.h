#ifndef HLTrigger_btau_HLTJetTag_h
#define HLTrigger_btau_HLTJetTag_h

/** \class HLTJetTag
 *
 *  This class is an HLTFilter (a specialized EDFilter) implementing
 *  tagged multi-jet trigger for b and tau.
 *  It should be run after the normal multi-jet trigger.
 *
 *
 *  \author Arnaud Gay, Ian Tomalin
 *  \maintainer Andrea Bocci
 *
 */

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

template <typename T>
class HLTJetTag : public HLTFilter {
public:
  explicit HLTJetTag(edm::ParameterSet const& config);
  ~HLTJetTag() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool hltFilter(edm::Event& event,
                 edm::EventSetup const& setup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  auto logTrace() const {
    auto const moduleName = moduleDescription().moduleName();
    return LogTrace(moduleName) << "[" << moduleName << "] ";
  }

  bool findTagValueByMinDeltaR2(float& jetTagValue,
                                T const& jet,
                                reco::JetTagCollection const& jetTags,
                                float maxDeltaR2) const;

  edm::InputTag const m_Jets;  // module label of input JetCollection
  edm::EDGetTokenT<std::vector<T>> const m_JetsToken;
  edm::InputTag const m_JetTags;  // module label of input JetTagCollection
  edm::EDGetTokenT<reco::JetTagCollection> const m_JetTagsToken;
  double const m_MinTag, m_MaxTag;               // tag discriminator cuts applied to each jet
  int const m_MinJets;                           // min. number of jets required to be tagged
  bool const m_MatchJetsByDeltaR;                // flag to enable jet matching by minimum Delta-R distance
  double const m_MaxJetDeltaR, m_MaxJetDeltaR2;  // max delta-R distance used in jet matching
  int const m_TriggerType;                       // type of trigger objects in output product
};

template <typename T>
HLTJetTag<T>::HLTJetTag(edm::ParameterSet const& config)
    : HLTFilter(config),
      m_Jets(config.getParameter<edm::InputTag>("Jets")),
      m_JetsToken(consumes(m_Jets)),
      m_JetTags(config.getParameter<edm::InputTag>("JetTags")),
      m_JetTagsToken(consumes(m_JetTags)),
      m_MinTag(config.getParameter<double>("MinTag")),
      m_MaxTag(config.getParameter<double>("MaxTag")),
      m_MinJets(config.getParameter<int>("MinJets")),
      m_MatchJetsByDeltaR(config.getParameter<bool>("MatchJetsByDeltaR")),
      m_MaxJetDeltaR(config.getParameter<double>("MaxJetDeltaR")),
      m_MaxJetDeltaR2(m_MaxJetDeltaR * m_MaxJetDeltaR),
      m_TriggerType(config.getParameter<int>("TriggerType")) {
  if (m_MatchJetsByDeltaR and m_MaxJetDeltaR < 0) {
    throw cms::Exception("InvalidConfigurationParameter")
        << "invalid value for parameter \"MaxJetDeltaR\" (must be >= 0): " << m_MaxJetDeltaR;
  }

  logTrace() << "HLTFilter parameters:\n"
             << "\ttype of jets used: " << m_Jets.encode() << "\n"
             << "\ttype of tagged jets used: " << m_JetTags.encode() << "\n"
             << "\tmin/max tag value: [" << m_MinTag << ", " << m_MaxTag << "]\n"
             << "\tmin num. of tagged jets: " << m_MinJets << "\n"
             << "\tmatch jets by Delta-R: " << m_MatchJetsByDeltaR << "\n"
             << "\tmax Delta-R for jet matching: " << m_MaxJetDeltaR << "\n"
             << "\tTriggerType: " << m_TriggerType << "\n";
}

template <typename T>
void HLTJetTag<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("Jets", edm::InputTag("hltJetCollection"));
  desc.add<edm::InputTag>("JetTags", edm::InputTag("hltJetTagCollection"));
  desc.add<double>("MinTag", 2.0);
  desc.add<double>("MaxTag", 999999.0);
  desc.add<int>("MinJets", 1);
  desc.add<bool>("MatchJetsByDeltaR", false);
  desc.add<double>("MaxJetDeltaR", 0.1);
  desc.add<int>("TriggerType", 0);
  descriptions.addWithDefaultLabel(desc);
}

template <typename T>
bool HLTJetTag<T>::findTagValueByMinDeltaR2(float& jetTagValue,
                                            T const& jet,
                                            reco::JetTagCollection const& jetTags,
                                            float maxDeltaR2) const {
  bool ret = false;
  for (auto const& jetTag : jetTags) {
    auto const tmpDR2 = reco::deltaR2(jet, *(jetTag.first));
    if (tmpDR2 < maxDeltaR2) {
      maxDeltaR2 = tmpDR2;
      jetTagValue = jetTag.second;
      ret = true;
    }
  }

  return ret;
}

template <typename T>
bool HLTJetTag<T>::hltFilter(edm::Event& event,
                             edm::EventSetup const& setup,
                             trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  auto const h_Jets = event.getHandle(m_JetsToken);
  if (saveTags())
    filterproduct.addCollectionTag(m_Jets);

  auto const h_JetTags = event.getHandle(m_JetTagsToken);

  // check if the product this one depends on is available
  auto const& handle = h_JetTags;
  auto const& dependent = handle->keyProduct();
  if (not dependent.isNull() and not dependent.hasCache()) {
    // only an empty AssociationVector can have an invalid dependent collection
    edm::StableProvenance const& dependent_provenance = event.getStableProvenance(dependent.id());
    if (dependent_provenance.branchDescription().dropped())
      // FIXME the error message should be made prettier
      throw cms::Exception("ProductNotFound") << "Product " << handle.provenance()->branchName() << " requires product "
                                              << dependent_provenance.branchName() << ", which has been dropped";
  }

  int nJetTags = 0;

  for (size_t iJet = 0; iJet < h_Jets->size(); ++iJet) {
    auto const& jet = (*h_Jets)[iJet];
    auto const jetRef = edm::Ref<std::vector<T>>(h_Jets, iJet);

    float jetTag = -1.f;
    if (m_MatchJetsByDeltaR) {
      // if match-by-DR is used, update jetTag by reference and use it only if the matching is valid
      if (not findTagValueByMinDeltaR2(jetTag, jet, *h_JetTags, m_MaxJetDeltaR2)) {
        continue;
      }
    } else {
      // operator[] checks consistency between h_Jets and h_JetTags
      jetTag = (*h_JetTags)[reco::JetBaseRef(jetRef)];
    }

    // Check if jet is tagged
    bool const jetIsTagged = (m_MinTag <= jetTag) and (jetTag <= m_MaxTag);

    if (jetIsTagged) {
      // Store a reference to the jets which passed tagging cuts
      filterproduct.addObject(m_TriggerType, jetRef);
      ++nJetTags;
    }

    logTrace() << "Matched Jets -- Jet[" << iJet << "]"
               << ": pt=" << jet.pt() << ", eta=" << jet.eta() << ", phi=" << jet.phi() << ", tag=" << jetTag
               << ", isTagged=" << jetIsTagged;
  }

  // filter decision
  bool const accept = (nJetTags >= m_MinJets);

  logTrace() << " trigger accept = " << accept << ", nJetTags = " << nJetTags;

  return accept;
}

#endif  // HLTrigger_btau_HLTJetTag_h
