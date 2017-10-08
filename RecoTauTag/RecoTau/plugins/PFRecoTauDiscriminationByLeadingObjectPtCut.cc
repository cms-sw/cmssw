#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

/* 
 * class PFRecoTauDiscriminationByLeadingObjectPtCut
 * created : October 08 2008,
 * revised : Wed Aug 19 17:13:04 PDT 2009
 * Authors : Simone Gennai (SNS), Evan Friis (UC Davis)
 */

using namespace reco;

class PFRecoTauDiscriminationByLeadingObjectPtCut : public PFTauDiscriminationProducerBase  {
   public:
      explicit PFRecoTauDiscriminationByLeadingObjectPtCut(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig){   
         chargedOnly_     = iConfig.getParameter<bool>("UseOnlyChargedHadrons");
         minPtLeadObject_ = iConfig.getParameter<double>("MinPtLeadingObject");
      }
      ~PFRecoTauDiscriminationByLeadingObjectPtCut() override{} 
      double discriminate(const PFTauRef& pfTau) const override;
   private:
      bool chargedOnly_;
      double minPtLeadObject_;
};

double PFRecoTauDiscriminationByLeadingObjectPtCut::discriminate(const PFTauRef& thePFTauRef) const
{
   double leadObjectPt = -1.;
   if( chargedOnly_ )
   {
      // consider only charged hadrons.  note that the leadPFChargedHadrCand is the highest pt
      // charged signal cone object above the quality cut level (typically 0.5 GeV).  
      if( thePFTauRef->leadPFChargedHadrCand().isNonnull() )
      {
         leadObjectPt = thePFTauRef->leadPFChargedHadrCand()->pt();
      }
   } 
   else
   {
      // If using the 'leading pion' option, require that:
      //   1) at least one charged hadron exists above threshold (thePFTauRef->leadPFChargedHadrCand().isNonnull())
      //   2) the lead PFCand exists.  In the case that the highest pt charged hadron is above the PFRecoTauProducer threshold 
      //      (typically 5 GeV), the leadPFCand and the leadPFChargedHadrCand are the same object.  If the leadPFChargedHadrCand
      //      is below 5GeV, but there exists a neutral PF particle > 5 GeV, it is set to be the leadPFCand
      if( thePFTauRef->leadPFCand().isNonnull() && thePFTauRef->leadPFChargedHadrCand().isNonnull() )
      {
         leadObjectPt = thePFTauRef->leadPFCand()->pt();
      }
   }

   return ( leadObjectPt > minPtLeadObject_ ? 1. : 0. );
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByLeadingObjectPtCut);

/*
void PFRecoTauDiscriminationByLeadingPionPtCut::produce(edm::Event& iEvent,const edm::EventSetup& iEventSetup){
   edm::Handle<PFTauCollection> thePFTauCollection;
   iEvent.getByLabel(PFTauProducer_,thePFTauCollection);


   auto thePFTauDiscriminatorByLeadingPionPtCut = std::make_unique<PFTauDiscriminator(PFTauRefProd>(thePFTauCollection));

   //loop over the PFTau candidates
   for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
      PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
      PFTau thePFTau=*thePFTauRef;
      double theleadTrackPtCutDiscriminator = 0.;   
      // fill the AssociationVector object
      if (!thePFTau.leadPFCand() || !thePFTau.leadPFChargedHadrCand()) 
      {
         theleadTrackPtCutDiscriminator=0.;
      }
      else if(thePFTau.leadPFCand()->pt() > minPtLeadTrack_) theleadTrackPtCutDiscriminator=1.;

      thePFTauDiscriminatorByLeadingPionPtCut->setValue(iPFTau,theleadTrackPtCutDiscriminator);
   }

   iEvent.put(std::move(thePFTauDiscriminatorByLeadingPionPtCut));

}
   
*/
