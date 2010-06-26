/* 
 * class PFRecoTauDecayModeValueExtractor
 * created : April 16, 2009
 * revised : ,
 * Authors : Evan K. Friis, (UC Davis), Simone Gennai (SNS)
 *
 * Associates an extracted value from the the PFTauDecayMode production 
 * to its underlying PFTau, in PFTau discriminator format.
 *
 * The following corrections are applied in the PFTauDecayMode production
 *
 * 1) UE filtering
 * 2) Merging of PFGammas into candidate pi zeros
 * 3) Application of mass hypothesis (charged/neutral pion) to constituents
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include <boost/shared_ptr.hpp>


class PFRecoTauDecayModeValueExtractor : public PFTauDiscriminationProducerBase  
{
   public:
      explicit PFRecoTauDecayModeValueExtractor(const ParameterSet& iConfig):
         PFTauDiscriminationProducerBase(iConfig),
         func_(new StringObjectFunction<reco::PFTauDecayMode>(iConfig.getParameter<string>("expression")))
      {   
         PFTauDecayModeProducer_     = iConfig.getParameter<InputTag>("PFTauDecayModeProducer");
      }
      ~PFRecoTauDecayModeValueExtractor(){} 
      double discriminate(const PFTauRef& pfTau);
      void beginEvent(const Event& event, const EventSetup& evtSetup);
   private:
      InputTag PFTauDecayModeProducer_;
      Handle<PFTauDecayModeAssociation> theDMAssoc;
      boost::shared_ptr<StringObjectFunction<reco::PFTauDecayMode> > func_;
};

void PFRecoTauDecayModeValueExtractor::beginEvent(const Event& event, const EventSetup& evtSetup)
{
   // Get the PFTau Decay Modes
   event.getByLabel(PFTauDecayModeProducer_, theDMAssoc);
}

double PFRecoTauDecayModeValueExtractor::discriminate(const PFTauRef& thePFTauRef)
{
   const PFTauDecayMode& theDecayMode = (*theDMAssoc)[thePFTauRef];
   return (*func_)(theDecayMode);
}

DEFINE_FWK_MODULE(PFRecoTauDecayModeValueExtractor);
