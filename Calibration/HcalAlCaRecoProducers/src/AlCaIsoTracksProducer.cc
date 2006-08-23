#include "Calibration/HcalAlCaRecoProducers/interface/AlCaIsoTracksProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


AlCaIsoTracksProducer::AlCaIsoTracksProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<reco::TrackCollection>();
}


AlCaIsoTracksProducer::~AlCaIsoTracksProducer()
{
 

}


// ------------ method called to produce the data  ------------
void
AlCaIsoTracksProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
    
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
            int isol = 1;
            for (reco::TrackCollection::const_iterator track1=track+1; track1!=tC.end(); track1++)
            {
               double dx = track->vertex().x()-track1->vertex().x();
               double dy = track->vertex().y()-track1->vertex().y();
               double dz = track->vertex().z()-track1->vertex().z();
               double drvert = sqrt(dx*dx+dy*dy+dz*dz);
               if(drvert > 0.005) continue;    
               double deta = track->momentum().eta() - track1->momentum().eta();  
               double dphi = fabs(track->momentum().phi() - track1->momentum().phi());
               if (dphi > atan(1.)*4.) dphi = 8.*atan(1.) - dphi;
               double ddir = sqrt(deta*deta-dphi*dphi);
               if( ddir < 0.5 ) isol = 0;     
            }
      if(track->pt() > 2.)
      {
         if( isol == 1 ) outputTColl->push_back(*track);
      }      
   }

  //Put selected information in the event
  iEvent.put( outputTColl, "IsoTracksCollection");
}
