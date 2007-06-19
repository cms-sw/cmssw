#include "RecoTracker/NuclearSeedGenerator/interface/NuclearInteractionFinder.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

NuclearInteractionFinder::NuclearInteractionFinder(const edm::EventSetup& es, const edm::ParameterSet& iConfig) :
ptMin(iConfig.getParameter<double>("ptMin")),
maxPrimaryHits(iConfig.getParameter<int>("maxPrimaryHits")),
rescaleErrorFactor(iConfig.getParameter<double>("rescaleErrorFactor")),
checkCompletedTrack(iConfig.getParameter<bool>("checkCompletedTrack"))
{
   edm::ESHandle<Propagator> prop;
   edm::ESHandle<TrajectoryStateUpdator> upd;
   edm::ESHandle<Chi2MeasurementEstimatorBase> est;
   edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
   edm::ESHandle<GeometricSearchTracker>       theGeomSearchTrackerHandle;

   es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",prop);
   es.get<TrackingComponentsRecord>().get("Chi2",est);
   es.get<CkfComponentsRecord>().get(measurementTrackerHandle);
   es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTrackerHandle );
   es.get<IdealMagneticFieldRecord>().get(theMagField);

   thePropagator = prop.product();
   theEstimator = est.product();
   theMeasurementTracker = measurementTrackerHandle.product();
   theLayerMeasurements = new LayerMeasurements(theMeasurementTracker);
   theGeomSearchTracker = theGeomSearchTrackerHandle.product();
   theNavigationSchool  = new SimpleNavigationSchool(theGeomSearchTracker,&(*theMagField));

   // set the correct navigation
   NavigationSetter setter( *theNavigationSchool);
   LogDebug("NuclearSeedGenerator") << "New NuclearInteractionFinder instance with parameters : \n"
                                        << "ptMin : " << ptMin << "\n"
                                        << "maxPrimaryHits : " << maxPrimaryHits << "\n"
                                        << "rescaleErrorFactor : " << rescaleErrorFactor << "\n"
                                        << "checkCompletedTrack : " << checkCompletedTrack << "\n";
   nuclTester = new NuclearTester(es, iConfig);

   currentSeed = new SeedFromNuclearInteraction(es, iConfig) ;
}
//----------------------------------------------------------------------
void NuclearInteractionFinder::setEvent(const edm::Event& event) const
{
   theMeasurementTracker->update(event);
}

//----------------------------------------------------------------------
NuclearInteractionFinder::~NuclearInteractionFinder() {
  delete theLayerMeasurements;
  delete theNavigationSchool;
  delete nuclTester;
  delete currentSeed;
  delete thePrimaryHelix;
}

//----------------------------------------------------------------------
bool  NuclearInteractionFinder::run(const Trajectory& traj) {

        // initialization
        nuclTester->reset();
        allSeeds.clear();

        if(traj.empty() || !traj.isValid()) return false;

        std::vector<TrajectoryMeasurement> measurements = traj.measurements();

        if(traj.direction()==alongMomentum)  {
                std::reverse(measurements.begin(), measurements.end());
        }

        std::vector<TrajectoryMeasurement>::const_iterator it_meas = measurements.begin();

        std::vector<double> ncompatibleHits;
        bool NIfound = false;

        // Loop on all the RecHits. 
        while(!NIfound)
         {
           if(it_meas == measurements.end()) break;

           nuclTester->push_back(*it_meas, findCompatibleMeasurements(*it_meas, rescaleErrorFactor));

           LogDebug("NuclearSeedGenerator") << "Number of compatible meas:" << (nuclTester->back()).size() << "\n"
                                                << "Mean distance between hits :" << nuclTester->meanHitDistance() << "\n"
                                                << "Mean distance between hits :" << nuclTester->meanEstimate() << "\n";

           // don't check track which reach the end of the tracker
           if( checkCompletedTrack==false && (nuclTester->back()).empty() ) break;

           if(nuclTester->isNuclearInteraction()) NIfound=true;

           ++it_meas;
         }

        if(NIfound) {
            LogDebug("NuclearSeedGenerator") << "NUCLEAR INTERACTION FOUND at index : " << nuclTester->nuclearIndex() << "\n";

            // Get correct parametrization of the helix of the primary track at the interaction point (to be used by improveCurrentSeed)
            definePrimaryHelix(measurements.begin()+nuclTester->nuclearIndex()-1);

            this->fillSeeds( nuclTester->goodTMPair());

            return true;
        }

    return false;
}
//----------------------------------------------------------------------
void NuclearInteractionFinder::definePrimaryHelix(std::vector<TrajectoryMeasurement>::const_iterator it_meas) {
    // This method uses the 3 last TM after the interaction point to calculate the helix parameters

    GlobalPoint pt[3];
    for(int i=0; i<3; i++) {
       pt[i] = (it_meas->updatedState()).globalParameters().position();
       it_meas++;
    }
    thePrimaryHelix = new TangentHelix( pt[0], pt[1], pt[2] );
}
//----------------------------------------------------------------------
std::vector<TrajectoryMeasurement>
NuclearInteractionFinder::findCompatibleMeasurements(const TM& lastMeas, double rescale) const
{
  TSOS currentState = lastMeas.updatedState();
  LogDebug("NuclearSeedGenerator") << "currentState :" << currentState << "\n";

  currentState.rescaleError(rescale);
  return findMeasurementsFromTSOS(currentState, lastMeas.recHit()->geographicalId());
}

//----------------------------------------------------------------------
std::vector<TrajectoryMeasurement>
NuclearInteractionFinder::findMeasurementsFromTSOS(const TSOS& currentState, DetId detid) const {

  using namespace std;
  int invalidHits = 0;
  vector<TM> result;
  const DetLayer* lastLayer = theGeomSearchTracker->detLayer( detid ); 
  vector<const DetLayer*> nl;

  if(lastLayer) { 
          nl = lastLayer->nextLayers( *currentState.freeState(), alongMomentum);
  }
  else {
      edm::LogError("NuclearInteractionFinder") << "In findCompatibleMeasurements : lastLayer not accessible";
      return result;
  }

  if (nl.empty()) {
      LogDebug("NuclearSeedGenerator") << "In findCompatibleMeasurements :  no compatible layer found";
      return result;
  }

  for (vector<const DetLayer*>::iterator il = nl.begin();
       il != nl.end(); il++) {
    vector<TM> tmp =
      theLayerMeasurements->measurements((**il),currentState, *thePropagator, *theEstimator);
    if ( !tmp.empty()) {
      if ( result.empty()) result = tmp;
      else {
        // keep one dummy TM at the end, skip the others
        result.insert( result.end()-invalidHits, tmp.begin(), tmp.end());
      }
      invalidHits++;
    }
  }

  // sort the final result, keep dummy measurements at the end
  if ( result.size() > 1) {
    sort( result.begin(), result.end()-invalidHits, TrajMeasLessEstim());
  }
  return result;
}

//----------------------------------------------------------------------
void NuclearInteractionFinder::fillSeeds( const std::pair<TrajectoryMeasurement, std::vector<TrajectoryMeasurement> >& tmPairs ) {
  // This method returns the seeds calculated by the class SeedsFromNuclearInteraction

            const TM& innerTM = tmPairs.first;
            const std::vector<TM>& outerTMs = tmPairs.second;

            // Loop on all outer TM 
            for(std::vector<TM>::const_iterator outtm = outerTMs.begin(); outtm!=outerTMs.end(); outtm++) {
               if((innerTM.recHit())->isValid() && (outtm->recHit())->isValid()) {
                     currentSeed->setMeasurements(innerTM.updatedState(), innerTM.recHit(), outtm->recHit());
                     allSeeds.push_back(*currentSeed);
                }
                else  LogDebug("NuclearSeedGenerator") << "The initial hits for seeding are invalid" << "\n";
             } 
             return;
}
//----------------------------------------------------------------------
void NuclearInteractionFinder::getPersistentSeeds( std::auto_ptr<TrajectorySeedCollection>& output ) {
   for(std::vector<SeedFromNuclearInteraction>::const_iterator it_seed = allSeeds.begin(); it_seed != allSeeds.end(); it_seed++) {
       if(it_seed->isValid()) {
           output->push_back( it_seed->TrajSeed() );
       }
       else LogDebug("NuclearSeedGenerator") << "The seed is invalid" << "\n";
   }
}
//----------------------------------------------------------------------
void NuclearInteractionFinder::improveSeeds() {
        std::vector<SeedFromNuclearInteraction> newSeedCollection;
        double rescaleFactor = 10;

        // loop on all actual seeds
        for(std::vector<SeedFromNuclearInteraction>::const_iterator it_seed = allSeeds.begin(); it_seed != allSeeds.end(); it_seed++) {

              // rescale outer TSOS of the seed
              TSOS currentState = it_seed->updatedTSOS();
              currentState.rescaleError(rescaleFactor);

              // find compatible TM in an outer layer
              std::vector<TM> thirdTMs = findMeasurementsFromTSOS(  currentState, it_seed->outerHitDetId() ); 

              int i=0;

              // loop on those new TMs
              for(std::vector<TM>::const_iterator tm = thirdTMs.begin(); tm!= thirdTMs.end(); tm++) {

                   if( ! tm->recHit()->isValid() ) continue;

                   // create new seeds collection using the circle equation
                   currentSeed->setMeasurements(*thePrimaryHelix, it_seed->initialTSOS(), it_seed->outerHit(), tm->recHit() );
                   newSeedCollection.push_back( *currentSeed );
              }
       }
       allSeeds.clear();
       allSeeds = newSeedCollection;
}     
