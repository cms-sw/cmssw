/* 
 * class CaloRecoTauDiscriminationByLeadingTrackPtCut
 * created : October 08 2008,
 * revised : ,
 * Authors : Simone Gennai (SNS), Evan Friis (UC Davis)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "DataFormats/TrackReco/interface/Track.h"

using namespace reco;

class CaloRecoTauDiscriminationByLeadingTrackPtCut : public CaloTauDiscriminationProducerBase {
   public:
      explicit CaloRecoTauDiscriminationByLeadingTrackPtCut(const edm::ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
         minPtLeadTrack_ = iConfig.getParameter<double>("MinPtLeadingTrack");
      }
      ~CaloRecoTauDiscriminationByLeadingTrackPtCut(){} 
      double discriminate(const CaloTauRef& theCaloTauRef);

   private:
      double minPtLeadTrack_;
};

double CaloRecoTauDiscriminationByLeadingTrackPtCut::discriminate(const CaloTauRef& theCaloTauRef)
{
   double leadTrackPt_ = -1;

   if( theCaloTauRef->leadTrack().isNonnull() )
   {
      leadTrackPt_ = theCaloTauRef->leadTrack()->pt();
   } 

   return ( (leadTrackPt_ > minPtLeadTrack_) ? 1. : 0. );
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByLeadingTrackPtCut);
   
/*
   edm::Handle<CaloTauCollection> theCaloTauCollection;
   iEvent.getByLabel(CaloTauProducer_,theCaloTauCollection);

   double theleadTrackPtCutDiscriminator = 0.;
   auto_ptr<CaloTauDiscriminator> theCaloTauDiscriminatorByLeadingTrackPtCut(new CaloTauDiscriminator(CaloTauRefProd(theCaloTauCollection)));

   //loop over the CaloTau candidates
   for(size_t iCaloTau=0;iCaloTau<theCaloTauCollection->size();++iCaloTau) {
      CaloTauRef theCaloTauRef(theCaloTauCollection,iCaloTau);
      CaloTau theCaloTau=*theCaloTauRef;

      // fill the AssociationVector object
      if (!theCaloTau.leadTrack()) 
         theleadTrackPtCutDiscriminator=0.;
      else if(theCaloTau.leadTrack()->pt() > minPtLeadTrack_) theleadTrackPtCutDiscriminator=1.;

      theCaloTauDiscriminatorByLeadingTrackPtCut->setValue(iCaloTau,theleadTrackPtCutDiscriminator);
   }

   iEvent.put(theCaloTauDiscriminatorByLeadingTrackPtCut);

}
   
*/
