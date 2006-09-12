#include "Calibration/HcalAlCaRecoProducers/interface/AlCaDiJetsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;

//namespace cms
//{

AlCaDiJetsProducer::AlCaDiJetsProducer(const edm::ParameterSet& iConfig):
       mInput(iConfig.getParameter<std::string>("src"))
{
   
   //register your products

   produces<reco::TrackCollection>();
   produces<CaloJetCollection>();
}


AlCaDiJetsProducer::~AlCaDiJetsProducer()
{
 

}


// ------------ method called to produce the data  ------------
void
AlCaDiJetsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
// Jet Collection
   edm::Handle<CaloJetCollection> jets;                        //Define Inputs
   iEvent.getByLabel(mInput, jets);                            //Get Inputs
   auto_ptr<CaloJetCollection> result (new CaloJetCollection); //Corrected jets
   double myvalue = 4.*atan(1.);
   double twomyvalue = 8.*atan(1.); 
   CaloJet fJet1,fJet2;
   if(jets->size() > 1 )
   {
     fJet1 = (*jets)[0];
     fJet2 = (*jets)[1];
     double phi1=fabs(fJet1.phi());
     double phi2=fabs(fJet2.phi());
     double dphi = fabs(phi1-phi2);
     if (dphi > myvalue) dphi = twomyvalue-dphi;
     double degreedphi = dphi*180./myvalue;
     if (fabs(degreedphi-180.) > 10. ) return;
     result->push_back (fJet1);
     result->push_back (fJet2);
   } else
   {
     return;
   }

// Track Collection    
   edm::Handle<reco::TrackCollection> trackCollection;

   try {
     iEvent.getByType(trackCollection);
   } catch ( std::exception& ex ) {
     LogDebug("") << "AlCaIsoTracksProducer: Error! can't get product!" << std::endl;
   }
   const reco::TrackCollection tC = *(trackCollection.product());
  
   //Create empty output collections

   std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);

   for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end()-1; track++)
   {
               
               double deta = track->momentum().eta() - fJet1.eta();  
               double dphi = fabs(track->momentum().phi() - fJet1.phi());
               if (dphi > atan(1.)*4.) dphi = 8.*atan(1.) - dphi;
               double ddir1 = sqrt(deta*deta-dphi*dphi);
               deta = track->momentum().eta() - fJet2.eta();
               dphi = fabs(track->momentum().phi() - fJet2.phi());
               if (dphi > atan(1.)*4.) dphi = 8.*atan(1.) - dphi;
               double ddir2 = sqrt(deta*deta-dphi*dphi);

      if( ddir1 < 0.5  || ddir2 < 0.5)      
      {
         outputTColl->push_back(*track);
      } 
   }
  //Put selected information in the event
  iEvent.put( outputTColl, "JetTracksCollection");
  iEvent.put( result, "DijetBackToBackCollection");
}
//}
