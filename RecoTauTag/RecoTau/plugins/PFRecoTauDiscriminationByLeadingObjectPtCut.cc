#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

/* 
 * class PFRecoTauDiscriminationByLeadingObjectPtCut
 * created : October 08 2008,
 * revised : Wed Aug 19 17:13:04 PDT 2009
 * Authors : Simone Gennai (SNS), Evan Friis (UC Davis)
 */

using namespace reco;

class PFRecoTauDiscriminationByLeadingObjectPtCut : public PFTauDiscriminationProducerBase {
public:
  explicit PFRecoTauDiscriminationByLeadingObjectPtCut(const edm::ParameterSet& iConfig)
      : PFTauDiscriminationProducerBase(iConfig) {
    chargedOnly_ = iConfig.getParameter<bool>("UseOnlyChargedHadrons");
    minPtLeadObject_ = iConfig.getParameter<double>("MinPtLeadingObject");
  }
  ~PFRecoTauDiscriminationByLeadingObjectPtCut() override {}
  double discriminate(const PFTauRef& pfTau) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool chargedOnly_;
  double minPtLeadObject_;
};

double PFRecoTauDiscriminationByLeadingObjectPtCut::discriminate(const PFTauRef& thePFTauRef) const {
  double leadObjectPt = -1.;
  if (chargedOnly_) {
    // consider only charged hadrons.  note that the leadChargedHadrCand is the highest pt
    // charged signal cone object above the quality cut level (typically 0.5 GeV).
    if (thePFTauRef->leadChargedHadrCand().isNonnull()) {
      leadObjectPt = thePFTauRef->leadChargedHadrCand()->pt();
    }
  } else {
    // If using the 'leading pion' option, require that:
    //   1) at least one charged hadron exists above threshold (thePFTauRef->leadChargedHadrCand().isNonnull())
    //   2) the lead PFCand exists.  In the case that the highest pt charged hadron is above the PFRecoTauProducer threshold
    //      (typically 5 GeV), the leadCand and the leadChargedHadrCand are the same object.  If the leadChargedHadrCand
    //      is below 5GeV, but there exists a neutral PF particle > 5 GeV, it is set to be the leadCand
    if (thePFTauRef->leadCand().isNonnull() && thePFTauRef->leadChargedHadrCand().isNonnull()) {
      leadObjectPt = thePFTauRef->leadCand()->pt();
    }
  }

  return (leadObjectPt > minPtLeadObject_ ? 1. : 0.);
}

void PFRecoTauDiscriminationByLeadingObjectPtCut::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationByLeadingObjectPtCut
  edm::ParameterSetDescription desc;
  desc.add<double>("MinPtLeadingObject", 5.0);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }
  desc.add<bool>("UseOnlyChargedHadrons", false);
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));
  descriptions.add("pfRecoTauDiscriminationByLeadingObjectPtCut", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByLeadingObjectPtCut);
