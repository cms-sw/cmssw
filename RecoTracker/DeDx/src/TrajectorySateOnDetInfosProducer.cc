// -*- C++ -*-
//
// Package:    TrajectorySateOnDetInfosProducer
// Class:      TrajectorySateOnDetInfosProducer
// 
/**\class TrajectorySateOnDetInfosProducer TrajectorySateOnDetInfosProducer.cc RecoTracker/TrajectorySateOnDetInfosProducer/src/TrajectorySateOnDetInfosProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  andrea
//         Created:  Thu May 31 14:09:02 CEST 2007
// $Id: TrajectorySateOnDetInfosProducer.cc,v 1.9 2007/08/13 08:01:07 arizzi Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"
#include "DataFormats/TrackReco/interface/TrackTrajectorySateOnDetInfos.h"
#include "RecoTracker/DeDx/interface/TrajectorySateOnDetInfosProducer.h"



using namespace reco;
using namespace std;
using namespace edm;


//typedef edm::AssociationMap<reco::TrackRefProd,reco::TrajectorySateOnDetInfoCollection > TrackTrajectorySateOnDetInfosCollection2;

//typedef  edm::OneToOne<reco::TrackRefProd,reco::TrajectorySateOnDetInfoCollection>  TrackTrajectorySateOnDetInfosCollection2;
//typedef edm::AssociationMap<edm::OneToOne<reco::TrackRefProd,reco::TrajectorySateOnDetInfoCollection,unsigned short> > TrackTrajectorySateOnDetInfosCollection2;
//typedef  edm::AssociationVector<reco::TrackRefProd,std::vector<reco::TrajectorySateOnDetInfoCollection> >  TrackTrajectorySateOnDetInfosCollection2;




TrajectorySateOnDetInfosProducer::TrajectorySateOnDetInfosProducer(const edm::ParameterSet& iConfig)
{
   m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");
   m_tracksTag                 = iConfig.getParameter<edm::InputTag>("Track");


   //register your products
   produces<reco::TrackTrajectorySateOnDetInfosCollection>();  
}


TrajectorySateOnDetInfosProducer::~TrajectorySateOnDetInfosProducer()
{ 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrajectorySateOnDetInfosProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
   iEvent.getByLabel(m_trajTrackAssociationTag, trajTrackAssociationHandle);
   const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

   edm::Handle<reco::TrackCollection> trackCollectionHandle;
   iEvent.getByLabel(m_tracksTag,trackCollectionHandle);
   const reco::TrackCollection &tracks=*trackCollectionHandle.product();

   TrackTrajectorySateOnDetInfosCollection* outputCollection = new TrackTrajectorySateOnDetInfosCollection(reco::TrackRefProd(trackCollectionHandle) );

   int track_index=0;
   for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); it++) {
      const Track      track = *it->val;   
      const Trajectory traj  = *it->key;

      TrajectorySateOnDetInfoCollection TSODI_Coll;
         
      vector<TrajectoryMeasurement> measurements = traj.measurements();
      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){

         TrajectoryStateOnSurface trajState              = measurement_it->updatedState();                     if( !trajState.isValid() ) continue;     
         const TrackingRecHit*         hit               = (*measurement_it->recHit()).hit();
         const SiStripRecHit2D*        sistripsimplehit  = dynamic_cast<const SiStripRecHit2D*>(hit);
         const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);

	 TrajectorySateOnDetInfo* TSODI_Temp;
         if(sistripsimplehit)
         {
	       TSODI_Temp = Get_TSODI(&traj, &trajState, sistripsimplehit);                  if(TSODI_Temp!=NULL)TSODI_Coll.push_back(*TSODI_Temp);
         }else if(sistripmatchedhit){
               TSODI_Temp = Get_TSODI(&traj, &trajState, sistripmatchedhit->monoHit()   );   if(TSODI_Temp!=NULL)TSODI_Coll.push_back(*TSODI_Temp);
               TSODI_Temp = Get_TSODI(&traj, &trajState, sistripmatchedhit->stereoHit() );   if(TSODI_Temp!=NULL)TSODI_Coll.push_back(*TSODI_Temp);
         }                 
      }
      outputCollection->setValue(track_index,TSODI_Coll);
      track_index++;
   }

   //put in the event the result
   std::auto_ptr<TrackTrajectorySateOnDetInfosCollection> outputs(outputCollection);
   iEvent.put(outputs);

}

TrajectorySateOnDetInfo* TrajectorySateOnDetInfosProducer::Get_TSODI(const Trajectory* traj, const TrajectoryStateOnSurface* trajSOS, const SiStripRecHit2D* hit)
{
   const LocalTrajectoryParameters   theLocalParameters = trajSOS->localParameters();
   const SiStripRecHit2D::ClusterRef theCluster         = hit->cluster();
   std::vector<float>                theLocalErrors;
   const AlgebraicSymMatrix55        theMatrix          = (trajSOS->localError()).matrix();
   theLocalErrors.push_back(theMatrix(0,0)); 
   theLocalErrors.push_back(theMatrix(0,1));
   theLocalErrors.push_back(theMatrix(0,2));
   theLocalErrors.push_back(theMatrix(0,3));
   theLocalErrors.push_back(theMatrix(0,4));
   theLocalErrors.push_back(theMatrix(1,0));
   theLocalErrors.push_back(theMatrix(1,1));
   theLocalErrors.push_back(theMatrix(1,2));
   theLocalErrors.push_back(theMatrix(1,3));
   theLocalErrors.push_back(theMatrix(1,4));
   theLocalErrors.push_back(theMatrix(2,0));
   theLocalErrors.push_back(theMatrix(2,1));
   theLocalErrors.push_back(theMatrix(2,2));
   theLocalErrors.push_back(theMatrix(2,3));
   theLocalErrors.push_back(theMatrix(2,4));
   theLocalErrors.push_back(theMatrix(3,0));
   theLocalErrors.push_back(theMatrix(3,1));
   theLocalErrors.push_back(theMatrix(3,2));
   theLocalErrors.push_back(theMatrix(3,3));
   theLocalErrors.push_back(theMatrix(3,4));
   theLocalErrors.push_back(theMatrix(4,0));
   theLocalErrors.push_back(theMatrix(4,1));
   theLocalErrors.push_back(theMatrix(4,2));
   theLocalErrors.push_back(theMatrix(4,3));
   theLocalErrors.push_back(theMatrix(4,4));
   
   TrajectorySateOnDetInfo* toReturn = new TrajectorySateOnDetInfo(theLocalParameters, theLocalErrors, theCluster);
   return toReturn;
}

// ------------ method called once each job just before starting event loop  ------------
void 
TrajectorySateOnDetInfosProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrajectorySateOnDetInfosProducer::endJob() {
//TODO: if verbose level very high, print the detid->calib map
}


//define this as a plug-in
DEFINE_FWK_MODULE(TrajectorySateOnDetInfosProducer);
