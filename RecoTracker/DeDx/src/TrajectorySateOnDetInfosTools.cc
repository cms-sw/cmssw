#include "RecoTracker/DeDx/interface/TrajectorySateOnDetInfosTools.h"


using namespace edm;
using namespace reco;
using namespace std;


TrajectorySateOnDetInfo* TSODI::Get_TSODI(const Trajectory* traj, const TrajectoryStateOnSurface* trajSOS, const SiStripRecHit2D* hit)
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


TrackTrajectorySateOnDetInfosCollection* TSODI::Get_TSODICollection(const TrajTrackAssociationCollection TrajToTrackMap, edm::Handle<reco::TrackCollection> trackCollectionHandle)
{
   TrackTrajectorySateOnDetInfosCollection* outputCollection = new TrackTrajectorySateOnDetInfosCollection(reco::TrackRefProd(trackCollectionHandle) );
//   TrackTrajectorySateOnDetInfosCollection* outputCollection = new TrackTrajectorySateOnDetInfosCollection();
//   int K = 0;
   int track_index=0;
   for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); it++) {
      track_index++;
      const Track      track = *it->val;
      const Trajectory traj  = *it->key;

/*      printf("%6.2f<%6.2f  -  %6.2f>%6.2f\n", track.p(),Track_PMin,track.chi2(),Track_Chi2Max);
      if(track.p()    < Track_PMin    )continue;
      if(track.p()    > Track_PMax    )continue;
      if(track.chi2() > Track_Chi2Max )continue;
      K++;

      printf("NoSkipped\n");
*/
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
               TSODI_Temp = TSODI::Get_TSODI(&traj, &trajState, sistripsimplehit);                  if(TSODI_Temp!=NULL)TSODI_Coll.push_back(*TSODI_Temp);
         }else if(sistripmatchedhit){
               TSODI_Temp = TSODI::Get_TSODI(&traj, &trajState, sistripmatchedhit->monoHit()   );   if(TSODI_Temp!=NULL)TSODI_Coll.push_back(*TSODI_Temp);
               TSODI_Temp = TSODI::Get_TSODI(&traj, &trajState, sistripmatchedhit->stereoHit() );   if(TSODI_Temp!=NULL)TSODI_Coll.push_back(*TSODI_Temp);
         }
      }
      outputCollection->setValue(track_index-1,TSODI_Coll);
//      outputCollection->insert(it->val,TSODI_Coll);
   }

//   printf("SIZE = %i >< %i\n",outputCollection->size(),K );
   return outputCollection;
}

