/* class CaloRecoTauDiscriminationByIsolation
 * created : Jul 23 2007,
 * revised : Sep 5 2007,
 * contributors : Ludovic Houchu (Ludovic.Houchu@cern.ch ; IPHC, Strasbourg), Christian Veelken (veelken@fnal.gov ; UC Davis), Evan Friis (UC Davis)
 */

#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

class CaloRecoTauDiscriminationByIsolation : public CaloTauDiscriminationProducerBase {
   public:
      explicit CaloRecoTauDiscriminationByIsolation(const ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
         ApplyDiscriminationByTrackerIsolation_ = iConfig.getParameter<bool>("ApplyDiscriminationByTrackerIsolation");
         TrackerIsolAnnulus_Tracksmaxn_         = iConfig.getParameter<int>("TrackerIsolAnnulus_Tracksmaxn");   
      }
      ~CaloRecoTauDiscriminationByIsolation(){} 
      double discriminate(const CaloTauRef& theCaloTauRef);
   private:  
      bool ApplyDiscriminationByTrackerIsolation_;
      int TrackerIsolAnnulus_Tracksmaxn_;   
};

double CaloRecoTauDiscriminationByIsolation::discriminate(const CaloTauRef& theCaloTauRef)
{
   CaloTau theCaloTau=*theCaloTauRef;
   math::XYZVector theCaloTau_XYZVector=theCaloTau.momentum();   
   CaloTauElementsOperators theCaloTauElementsOperators(theCaloTau); 	

   if (ApplyDiscriminationByTrackerIsolation_){  
      // optional selection by a tracker isolation : ask for 0 reco::Track in an isolation annulus around a leading reco::Track axis
      double theTrackerIsolationDiscriminator;
      theTrackerIsolationDiscriminator=theCaloTauElementsOperators.discriminatorByIsolTracksN(TrackerIsolAnnulus_Tracksmaxn_);
      if (theTrackerIsolationDiscriminator==0){
         return 0.;
      }
   }
   // N.B. the lead track requirement must be included in the discriminants
   return 1.;
}
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByIsolation);

   /*
void CaloRecoTauDiscriminationByIsolation::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<CaloTauCollection> theCaloTauCollection;
  iEvent.getByLabel(CaloTauProducer_,theCaloTauCollection);

  // fill the AssociationVector object
  auto_ptr<CaloTauDiscriminator> theCaloTauDiscriminatorByIsolation(new CaloTauDiscriminator(CaloTauRefProd(theCaloTauCollection)));

  for(size_t iCaloTau=0;iCaloTau<theCaloTauCollection->size();++iCaloTau) {
    CaloTauRef theCaloTauRef(theCaloTauCollection,iCaloTau);
    CaloTau theCaloTau=*theCaloTauRef;
    math::XYZVector theCaloTau_XYZVector=theCaloTau.momentum();   
    CaloTauElementsOperators theCaloTauElementsOperators(theCaloTau); 	
    
    if (ApplyDiscriminationByTrackerIsolation_){  
      // optional selection by a tracker isolation : ask for 0 reco::Track in an isolation annulus around a leading reco::Track axis
      double theTrackerIsolationDiscriminator;
      theTrackerIsolationDiscriminator=theCaloTauElementsOperators.discriminatorByIsolTracksN(TrackerIsolAnnulus_Tracksmaxn_);
      if (theTrackerIsolationDiscriminator==0){
	theCaloTauDiscriminatorByIsolation->setValue(iCaloTau,0);
	continue;
      }
    }
    
    // not optional selection : ask for a leading (Pt>minPt) reco::Track in a matching cone around the CaloJet axis
    double theleadTkDiscriminator=NAN;
    if (!theCaloTau.leadTrack()) theleadTkDiscriminator=0;
    if (theleadTkDiscriminator==0) theCaloTauDiscriminatorByIsolation->setValue(iCaloTau,0);
    else theCaloTauDiscriminatorByIsolation->setValue(iCaloTau,1);
  }
  
  iEvent.put(theCaloTauDiscriminatorByIsolation);
}
*/
