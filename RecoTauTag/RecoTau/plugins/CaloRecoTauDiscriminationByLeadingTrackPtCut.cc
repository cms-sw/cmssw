/* 
 * class CaloRecoTauDiscriminationByLeadingTrackPtCut
 * created : October 08 2008,
 * revised : ,
 * Authors : Simone Gennai (SNS), Evan Friis (UC Davis)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "DataFormats/TrackReco/interface/Track.h"

using namespace reco;

class CaloRecoTauDiscriminationByLeadingTrackPtCut final : public CaloTauDiscriminationProducerBase {
   public:
      explicit CaloRecoTauDiscriminationByLeadingTrackPtCut(const edm::ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
         minPtLeadTrack_ = iConfig.getParameter<double>("MinPtLeadingTrack");
      }
      ~CaloRecoTauDiscriminationByLeadingTrackPtCut() override{} 
      double discriminate(const CaloTauRef& theCaloTauRef) const override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      double minPtLeadTrack_;
};

double CaloRecoTauDiscriminationByLeadingTrackPtCut::discriminate(const CaloTauRef& theCaloTauRef) const
{
   double leadTrackPt_ = -1;

   if( theCaloTauRef->leadTrack().isNonnull() )
   {
      leadTrackPt_ = theCaloTauRef->leadTrack()->pt();
   } 

   return ( (leadTrackPt_ > minPtLeadTrack_) ? 1. : 0. );
}

void
CaloRecoTauDiscriminationByLeadingTrackPtCut::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // caloRecoTauDiscriminationByLeadingTrackPtCut
  edm::ParameterSetDescription desc;
  desc.add<double>("MinPtLeadingTrack", 5.0);
  desc.add<edm::InputTag>("CaloTauProducer", edm::InputTag("caloRecoTauProducer"));
  {
    edm::ParameterSetDescription requireLeadTrackCalo;
    requireLeadTrackCalo.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription leadTrack;
      leadTrack.add<double>("cut");
      leadTrack.add<edm::InputTag>("Producer");
      requireLeadTrackCalo.add<edm::ParameterSetDescription>("leadTrack", leadTrack);
    }
    desc.add<edm::ParameterSetDescription>("Prediscriminants", requireLeadTrackCalo);
  }
  descriptions.add("caloRecoTauDiscriminationByLeadingTrackPtCut", desc);
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByLeadingTrackPtCut);
   
/*
   edm::Handle<CaloTauCollection> theCaloTauCollection;
   iEvent.getByLabel(CaloTauProducer_,theCaloTauCollection);

   double theleadTrackPtCutDiscriminator = 0.;
   auto theCaloTauDiscriminatorByLeadingTrackPtCut = std::make_unique<CaloTauDiscriminator>(CaloTauRefProd(theCaloTauCollection));

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

   iEvent.put(std::move(theCaloTauDiscriminatorByLeadingTrackPtCut));

}
   
*/
