#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

/* class PFRecoTauDiscriminationByCharge
 *
 *  Discriminates taus by a |charge| == 1 requirement, 
 *  and optionally nTracks == 1 || 3 
 *
 * revised : Mon Aug 31 12:59:50 PDT 2009
 * Authors : Michele Pioppi, Evan Friis (UC Davis)
 */

using namespace reco;

class PFRecoTauDiscriminationByCharge : public PFTauDiscriminationProducerBase  {
   public:
      explicit PFRecoTauDiscriminationByCharge(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig){   
         chargeReq_        = iConfig.getParameter<uint32_t>("AbsChargeReq");
         oneOrThreeProng_  = iConfig.getParameter<bool>("ApplyOneOrThreeProngCut");
      }
      ~PFRecoTauDiscriminationByCharge() override{} 
      double discriminate(const PFTauRef& pfTau) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
   private:
      uint32_t chargeReq_;
      bool oneOrThreeProng_;
};

double PFRecoTauDiscriminationByCharge::discriminate(const PFTauRef& thePFTauRef) const
{
   uint16_t nSigTk        =  thePFTauRef->signalPFChargedHadrCands().size();
   bool chargeok          =  (std::abs(thePFTauRef->charge()) == int(chargeReq_));
   bool oneOrThreeProngOK =  ( (nSigTk==1) || (nSigTk==3) || !oneOrThreeProng_ );

   return ( (chargeok && oneOrThreeProngOK) ? 1. : 0. );
}

void
PFRecoTauDiscriminationByCharge::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationByCharge
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("AbsChargeReq", 1);
  desc.add<bool>("ApplyOneOrThreeProngCut", false);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducerHighEfficiency"));
  descriptions.add("pfRecoTauDiscriminationByCharge", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByCharge);
