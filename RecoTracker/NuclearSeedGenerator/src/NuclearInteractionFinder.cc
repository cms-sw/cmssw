#include "RecoTracker/NuclearSeedGenerator/interface/NuclearInteractionFinder.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h" 

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "FWCore/Framework/interface/ESHandle.h"


NuclearInteractionFinder::NuclearInteractionFinder(const edm::EventSetup& es, const edm::ParameterSet& iConfig) :
maxHits(iConfig.getParameter<int>("maxHits")),
rescaleErrorFactor(iConfig.getParameter<double>("rescaleErrorFactor")),
checkCompletedTrack(iConfig.getParameter<bool>("checkCompletedTrack")),
navigationSchoolName(iConfig.getParameter<std::string>("NavigationSchool"))
{

   std::string measurementTrackerName = iConfig.getParameter<std::string>("MeasurementTrackerName");

   edm::ESHandle<Propagator> prop;
   edm::ESHandle<TrajectoryStateUpdator> upd;
   edm::ESHandle<Chi2MeasurementEstimatorBase> est;
   edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
   edm::ESHandle<GeometricSearchTracker>       theGeomSearchTrackerHandle;
   edm::ESHandle<TrackerGeometry>              theTrackerGeom;

   es.get<TrackerDigiGeometryRecord> ().get (theTrackerGeom);
   es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",prop);
   es.get<TrackingComponentsRecord>().get("Chi2",est);
   es.get<CkfComponentsRecord>().get(measurementTrackerName, measurementTrackerHandle);
   es.get<TrackerRecoGeometryRecord>().get(theGeomSearchTrackerHandle );
   es.get<IdealMagneticFieldRecord>().get(theMagField);

   edm::ESHandle<NavigationSchool>             nav;
   es.get<NavigationSchoolRecord>().get(navigationSchoolName, nav);
   theNavigationSchool  = nav.product();

   thePropagator = prop.product();
   theEstimator = est.product();
   theMeasurementTracker = measurementTrackerHandle.product();
   theLayerMeasurements = new LayerMeasurements(theMeasurementTracker);
   theGeomSearchTracker = theGeomSearchTrackerHandle.product();

   LogDebug("NuclearSeedGenerator") << "New NuclearInteractionFinder instance with parameters : \n"
                                        << "maxHits : " << maxHits << "\n"
                                        << "rescaleErrorFactor : " << rescaleErrorFactor << "\n"
                                        << "checkCompletedTrack : " << checkCompletedTrack << "\n";

   nuclTester = new NuclearTester(maxHits, theEstimator, theTrackerGeom.product() );

   currentSeed = new SeedFromNuclearInteraction(thePropagator, theTrackerGeom.product(), iConfig) ;

   thePrimaryHelix = new TangentHelix();
}
//----------------------------------------------------------------------
void NuclearInteractionFinder::setEvent(const edm::Event& event) const
{
   theMeasurementTracker->update(event);
}

//----------------------------------------------------------------------
NuclearInteractionFinder::~NuclearInteractionFinder() {
  delete theLayerMeasurements;
  delete nuclTester;
  delete currentSeed;
  delete thePrimaryHelix;
}

//----------------------------------------------------------------------
bool  NuclearInteractionFinder::run(const Trajectory& traj) {

        if(traj.empty() || !traj.isValid()) return false;

        LogDebug("NuclearSeedGenerator") << "Analyzis of a new trajectory with a number of valid hits = " << traj.foundHits();

        std::vector<TrajectoryMeasurement> measurements = traj.measurements();

        // initialization
        nuclTester->reset( measurements.size() );
        allSeeds.clear();


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

	   // check only the maxHits outermost hits of the primary track
	   if(nuclTester->nHitsChecked() > maxHits) break;

           nuclTester->push_back(*it_meas, findCompatibleMeasurements(*it_meas, rescaleErrorFactor));

           LogDebug("NuclearSeedGenerator") << "Number of compatible meas:" << (nuclTester->back()).size() << "\n"
                                                << "Mean distance between hits :" << nuclTester->meanHitDistance() << "\n"
                                                << "Forward estimate :" << nuclTester->fwdEstimate() << "\n";

           // don't check tracks which reach the end of the tracker if checkCompletedTrack==false
           if( checkCompletedTrack==false && (nuclTester->compatibleHits()).front()==0 ) break;

           if(nuclTester->isNuclearInteraction()) NIfound=true;

           ++it_meas;
         }

        if(NIfound) {
            LogDebug("NuclearSeedGenerator") << "NUCLEAR INTERACTION FOUND at index : " << nuclTester->nuclearIndex()  << "\n";

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
    delete thePrimaryHelix;
    thePrimaryHelix = new TangentHelix( pt[0], pt[1], pt[2] );
}
//----------------------------------------------------------------------
std::vector<TrajectoryMeasurement>
NuclearInteractionFinder::findCompatibleMeasurements(const TM& lastMeas, double rescale) const
{
  TSOS currentState = lastMeas.updatedState();
  LogDebug("NuclearSeedGenerator") << "currentState :" << currentState << "\n";

  TSOS newState = rescaleError(rescale, currentState);
  return findMeasurementsFromTSOS(newState, lastMeas.recHit()->geographicalId());
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
std::auto_ptr<TrajectorySeedCollection> NuclearInteractionFinder::getPersistentSeeds() {
   std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
   for(std::vector<SeedFromNuclearInteraction>::const_iterator it_seed = allSeeds.begin(); it_seed != allSeeds.end(); it_seed++) {
       if(it_seed->isValid()) {
           output->push_back( it_seed->TrajSeed() );
       }
       else LogDebug("NuclearSeedGenerator") << "The seed is invalid" << "\n";
   }
   return output;
}
//----------------------------------------------------------------------
void NuclearInteractionFinder::improveSeeds() {
        std::vector<SeedFromNuclearInteraction> newSeedCollection;

        // loop on all actual seeds
        for(std::vector<SeedFromNuclearInteraction>::const_iterator it_seed = allSeeds.begin(); it_seed != allSeeds.end(); it_seed++) {

	      if( !it_seed->isValid() ) continue;

              // find compatible TM in an outer layer
              std::vector<TM> thirdTMs = findMeasurementsFromTSOS( it_seed->updatedTSOS() , it_seed->outerHitDetId() );

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
//----------------------------------------------------------------------
TrajectoryStateOnSurface NuclearInteractionFinder::rescaleError(float rescale, const TSOS& state) const {

     AlgebraicSymMatrix55 m(state.localError().matrix());
     AlgebraicSymMatrix55 mr;
     LocalTrajectoryParameters ltp = state.localParameters();

     // we assume that the error on q/p is equal to 20% of q/p * rescale 
     mr(0,0) = (ltp.signedInverseMomentum()*0.2*rescale)*(ltp.signedInverseMomentum()*0.2*rescale);

     // the error on dx/z and dy/dz is fixed to 10% (* rescale)
     mr(1,1) = 1E-2*rescale*rescale;
     mr(2,2) = 1E-2*rescale*rescale;

     // the error on the local x and y positions are not modified.
     mr(3,3) = m(3,3);
     mr(4,4) = m(4,4);

     return TSOS(ltp, mr, state.surface(), &(state.globalParameters().magneticField()), state.surfaceSide());
}
