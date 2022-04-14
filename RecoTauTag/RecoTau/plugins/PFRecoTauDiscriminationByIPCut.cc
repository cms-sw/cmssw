#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
/* 
 * class PFRecoTauDiscriminationByIPCut
 */

using namespace reco;

class PFRecoTauDiscriminationByIPCut : public PFTauDiscriminationProducerBase {
public:
  explicit PFRecoTauDiscriminationByIPCut(const edm::ParameterSet& iConfig)
      : PFTauDiscriminationProducerBase(iConfig),
        tausTIPToken_(consumes<PFTauTIPAssociationByRef>(iConfig.getParameter<edm::InputTag>("tausTIP"))),
        tauTIPSelectorString_(iConfig.getParameter<std::string>("cut")),
        tauTIPSelector_(tauTIPSelectorString_) {}
  ~PFRecoTauDiscriminationByIPCut() override {}
  void beginEvent(const edm::Event&, const edm::EventSetup&) override;
  double discriminate(const PFTauRef& pfTau) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> >
      PFTauTIPAssociationByRef;
  edm::EDGetTokenT<PFTauTIPAssociationByRef> tausTIPToken_;
  edm::Handle<PFTauTIPAssociationByRef> tausTIP_;

  const std::string tauTIPSelectorString_;
  const StringCutObjectSelector<reco::PFTauTransverseImpactParameter> tauTIPSelector_;
};

double PFRecoTauDiscriminationByIPCut::discriminate(const PFTauRef& thePFTauRef) const {
  return tauTIPSelector_(*(*tausTIP_)[thePFTauRef]) ? 1. : 0.;
}

void PFRecoTauDiscriminationByIPCut::beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup) {
  event.getByToken(tausTIPToken_, tausTIP_);
}

void PFRecoTauDiscriminationByIPCut::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tausTIP", edm::InputTag("hltTauIPCollection"));
  desc.add<std::string>("cut", "abs(dxy) > -999.");
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByIPCut);
