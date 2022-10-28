/** \class HLTPFTauIPFilter
 *
 *  This class is an HLTFilter able to implement
 *  generic cuts on the tau IP variables available in the PFTauTransverseImpactParameter collection
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include <vector>

class HLTPFTauIPFilter : public HLTFilter {
public:
  explicit HLTPFTauIPFilter(const edm::ParameterSet& config);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& setup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  const edm::InputTag m_TausInputTag;
  const edm::EDGetTokenT<reco::PFTauCollection> m_TausToken;
  const edm::EDGetTokenT<edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>>
      m_TausTIPToken;

  const int m_MinTaus;  // min. number of taus required to pass the cuts
  const std::string m_tauTIPSelectorString;
  const StringCutObjectSelector<reco::PFTauTransverseImpactParameter> m_tauTIPSelector;
  const int m_TriggerType;
};

HLTPFTauIPFilter::HLTPFTauIPFilter(const edm::ParameterSet& config)
    : HLTFilter(config),
      m_TausInputTag(config.getParameter<edm::InputTag>("Taus")),
      m_TausToken(consumes(m_TausInputTag)),
      m_TausTIPToken(consumes(config.getParameter<edm::InputTag>("TausIP"))),
      m_MinTaus(config.getParameter<int>("MinN")),
      m_tauTIPSelectorString(config.getParameter<std::string>("Cut")),
      m_tauTIPSelector(m_tauTIPSelectorString),
      m_TriggerType(config.getParameter<int>("TriggerType")) {
  edm::LogInfo("") << " (HLTPFTauIPFilter) trigger cuts:\n"
                   << " cut string value: [" << m_tauTIPSelectorString << "]\n"
                   << " min no. passing taus: " << m_MinTaus << ", TriggerType: " << m_TriggerType;
}

void HLTPFTauIPFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("Taus", edm::InputTag("hltTauCollection"));
  desc.add<edm::InputTag>("TausIP", edm::InputTag("hltTauIPCollection"));
  desc.add<int>("MinN", 1);
  desc.add<int>("TriggerType", 84);
  desc.add<std::string>("Cut", "dxy > -999.");
  descriptions.addWithDefaultLabel(desc);
}

bool HLTPFTauIPFilter::hltFilter(edm::Event& event,
                                 const edm::EventSetup& setup,
                                 trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  auto const h_Taus = event.getHandle(m_TausToken);
  if (saveTags())
    filterproduct.addCollectionTag(m_TausInputTag);

  auto const& TausTIP = event.get(m_TausTIPToken);

  int nTau = 0;
  for (reco::PFTauCollection::size_type iPFTau = 0; iPFTau < h_Taus->size(); iPFTau++) {
    reco::PFTauRef tauref(h_Taus, iPFTau);

    if (m_tauTIPSelector(*TausTIP[tauref])) {
      ++nTau;
      filterproduct.addObject(m_TriggerType, tauref);
    }
  }

  // filter decision
  bool const accept = (nTau >= m_MinTaus);
  edm::LogInfo("") << " trigger accept ? = " << accept << " nTau = " << nTau;

  return accept;
}

DEFINE_FWK_MODULE(HLTPFTauIPFilter);
