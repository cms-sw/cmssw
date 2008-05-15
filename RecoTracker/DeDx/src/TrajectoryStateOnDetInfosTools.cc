#include "RecoTracker/DeDx/interface/TrajectoryStateOnDetInfosTools.h"


using namespace edm;
using namespace reco;
using namespace std;

/*
TrajectoryStateOnDetInfo* TSODI::Get_TSODI(const TrajectoryStateOnSurface* trajSOS, TrackingRecHitRef theHit)
{
   const LocalTrajectoryParameters   theLocalParameters = trajSOS->localParameters();
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

  toReturn = new TrajectoryStateOnDetInfo(theLocalParameters, theLocalErrors, theHit);
   return toReturn;
}
*/

//TrackTrajectoryStateOnDetInfosCollection* TSODI::Fill_TSODICollection(const TrajTrackAssociationCollection TrajToTrackMap, edm::Handle<reco::TrackCollection> trackCollectionHandle)
//std::auto_ptr<TrackTrajectoryStateOnDetInfosCollection> TSODI::Fill_TSODICollection(const TrajTrackAssociationCollection TrajToTrackMap, edm::Handle<reco::TrackCollection> trackCollectionHandle)
void TSODI::Fill_TSODICollection(const TrajTrackAssociationCollection& TrajToTrackMap, TrackTrajectoryStateOnDetInfosCollection* outputCollection)
{
//   TrackTrajectoryStateOnDetInfosCollection* outputCollection = new TrackTrajectoryStateOnDetInfosCollection(reco::TrackRefProd(trackCollectionHandle) );
   int track_index=0;
   for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); it++) {
      track_index++;
      const Track      track = *it->val;
      const Trajectory traj  = *it->key;

      TrajectoryStateOnDetInfoCollection TSODI_Coll;

      vector<TrajectoryMeasurement> measurements = traj.measurements();
      for(unsigned int h=0;h<track.recHitsSize();h++){
         TrackingRecHitRef recHit = track.recHit(h);
         for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){

            TrajectoryStateOnSurface trajState              = measurement_it->updatedState();                     if( !trajState.isValid() ) continue;
            const TrackingRecHit*         hit               = (*measurement_it->recHit()).hit();

            if(recHit->geographicalId() != hit->geographicalId())continue;


            const LocalTrajectoryParameters   theLocalParameters = trajState.localParameters();
            std::vector<float>                theLocalErrors;
            const AlgebraicSymMatrix55        theMatrix          = (trajState.localError()).matrix();
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
            TSODI_Coll.push_back(TrajectoryStateOnDetInfo(theLocalParameters, theLocalErrors, recHit));

/*
            TrajectoryStateOnDetInfo* TSODI_Temp = TSODI::Get_TSODI(&trajState, recHit);
            if(TSODI_Temp!=NULL){
               TSODI_Coll.push_back(*TSODI_Temp);
	       delete;
            }            
*/
         }
      }
      outputCollection->setValue(track_index-1,TSODI_Coll);
   }

//   return outputCollection;
//   std::auto_ptr<TrackTrajectoryStateOnDetInfosCollection> to_return(outputCollection);
//   return to_return;
     return;
}


int TSODI::clusterSize(TrajectoryStateOnDetInfo* Tsodi, bool stereo)
{
   const TrackingRecHit*         hit               = (Tsodi->recHit()).get();
   const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
   const SiStripRecHit2D*        sistripsimplehit  = dynamic_cast<const SiStripRecHit2D*>       (hit);

   if(sistripsimplehit || sistripmatchedhit){
      if(sistripmatchedhit &&!stereo) sistripsimplehit = sistripmatchedhit->monoHit();
      if(sistripmatchedhit && stereo) sistripsimplehit = sistripmatchedhit->stereoHit();

      return ((sistripsimplehit->cluster()).get())->amplitudes().size();
   }else{
      return -1;
   }
   return -1;
}


int TSODI::charge(TrajectoryStateOnDetInfo* Tsodi, bool stereo)
{
   const TrackingRecHit*         hit               = (Tsodi->recHit()).get();
   const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
   const SiStripRecHit2D*        sistripsimplehit  = dynamic_cast<const SiStripRecHit2D*>       (hit);

   if(sistripsimplehit || sistripmatchedhit){
      if(sistripmatchedhit &&!stereo) sistripsimplehit = sistripmatchedhit->monoHit();
      if(sistripmatchedhit && stereo) sistripsimplehit = sistripmatchedhit->stereoHit();

      const SiStripCluster*   Cluster = (sistripsimplehit->cluster()).get();
      const vector<uint8_t >& Ampls   = Cluster->amplitudes();
//      const vector<uint16_t>& Ampls   = Cluster->amplitudes();

      unsigned int charge=0;
      for(unsigned int a=0;a<Ampls.size();a++){charge+=Ampls[a];}
      return charge;    
   }else{         
      return -1;
   }
   return -1;
}


double TSODI::thickness(TrajectoryStateOnDetInfo* Tsodi, edm::ESHandle<TrackerGeometry> tkGeom)
{
   const TrackingRecHit*         hit               = (Tsodi->recHit()).get();
   if(hit->type()>=1)return -1;

   const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(hit->geographicalId()));
   if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
//     std::cout << "this detID doesn't seem to belong to the Tracker" << std::endl;
     return -1;
   }

   return it->surface().bounds().thickness();
}

double TSODI::pathLength(TrajectoryStateOnDetInfo* Tsodi, edm::ESHandle<TrackerGeometry> tkGeom)
{
   double Thickness = TSODI::thickness(Tsodi, tkGeom);
   if(Thickness<0)return -1;
   double Cosine    = fabs(Tsodi->momentum().z() / Tsodi->momentum().mag());
   return (10.0*Thickness)/Cosine;
}


float TSODI::chargeOverPath(TrajectoryStateOnDetInfo* Tsodi, edm::ESHandle<TrackerGeometry> tkGeom){
   double Charge    = (double) TSODI::charge(Tsodi);
   if(charge<0)return -1;
   double pathLength = TSODI::pathLength(Tsodi, tkGeom);
   if(pathLength<0)return -1;

   return (float)(Charge/pathLength);
}


