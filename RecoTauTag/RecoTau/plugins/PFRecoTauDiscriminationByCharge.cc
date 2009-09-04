#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

/* class PFRecoTauDiscriminationByCharge
 *
 *  Discriminates taus by a |charge| == 1 requirement, 
 *  and optionally nTracks == 1 || 3 
 *
 * revised : Mon Aug 31 12:59:50 PDT 2009
 * Authors : Michele Pioppi, Evan Friis (UC Davis)
 */

class PFRecoTauDiscriminationByCharge : public PFTauDiscriminationProducerBase  {
   public:
      explicit PFRecoTauDiscriminationByCharge(const ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig){   
         chargeReq_        = iConfig.getParameter<uint32_t>("AbsChargeReq");
         oneOrThreeProng_  = iConfig.getParameter<bool>("ApplyOneOrThreeProngCut");
      }
      ~PFRecoTauDiscriminationByCharge(){} 
      double discriminate(const PFTauRef& pfTau);
   private:
      uint32_t chargeReq_;
      bool oneOrThreeProng_;
};

double PFRecoTauDiscriminationByCharge::discriminate(const PFTauRef& thePFTauRef)
{
   uint16_t nSigTk        =  thePFTauRef->signalPFChargedHadrCands().size();
   bool chargeok          =  (abs(thePFTauRef->charge()) == chargeReq_);
   bool oneOrThreeProngOK =  ( (nSigTk==1) || (nSigTk==3) || !oneOrThreeProng_ );

   return ( (chargeok && oneOrThreeProngOK) ? 1. : 0. );
}
DEFINE_FWK_MODULE(PFRecoTauDiscriminationByCharge);
