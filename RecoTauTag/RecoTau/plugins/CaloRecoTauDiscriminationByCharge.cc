#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

/* class CaloRecoTauDiscriminationByCharge
 *
 *  Discriminates taus by a |charge| == 1 requirement,
 *  and optionally nTracks == 1 || 3
 *
 * Authors : S.Lehti, copied from PFRecoTauDiscriminationByCharge
 */

namespace {

class CaloRecoTauDiscriminationByCharge final : public CaloTauDiscriminationProducerBase  {
  public:
    explicit CaloRecoTauDiscriminationByCharge(const edm::ParameterSet& iConfig)
        :CaloTauDiscriminationProducerBase(iConfig){
          chargeReq_        = iConfig.getParameter<uint32_t>("AbsChargeReq");
          oneOrThreeProng_  =
              iConfig.getParameter<bool>("ApplyOneOrThreeProngCut");
        }
    ~CaloRecoTauDiscriminationByCharge() override{}
    double discriminate(const reco::CaloTauRef& pfTau) const override;
  private:
    uint32_t chargeReq_;
    bool oneOrThreeProng_;
};

double CaloRecoTauDiscriminationByCharge::discriminate(
    const reco::CaloTauRef& theTauRef) const {
  uint16_t nSigTk =  theTauRef->signalTracks().size();
  bool chargeok = (abs(theTauRef->charge()) == int(chargeReq_));
  bool oneOrThreeProngOK =  ( (nSigTk==1) || (nSigTk==3) || !oneOrThreeProng_ );

  return ( (chargeok && oneOrThreeProngOK) ? 1. : 0. );
}

}
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByCharge);
