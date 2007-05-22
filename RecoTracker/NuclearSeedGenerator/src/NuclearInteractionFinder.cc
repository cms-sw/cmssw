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
   LogDebug("NuclearInteractionFinder") << "New NuclearInteractionFinder instance with parameters : \n"
                                        << "ptMin : " << ptMin << "\n"
                                        << "maxPrimaryHits : " << maxPrimaryHits << "\n"
                                        << "rescaleErrorFactor : " << rescaleErrorFactor << "\n"
                                        << "checkCompletedTrack : " << checkCompletedTrack << "\n";
   nuclTester = new NuclearTester(es, iConfig);

   theSeed = boost::shared_ptr<SeedFromNuclearInteraction>(new SeedFromNuclearInteraction(es, iConfig));
}
//----------------------------------------------------------------------
void NuclearInteractionFinder::setEvent(const edm::Event& event) const
{
   theMeasurementTracker->update(event);
}

//----------------------------------------------------------------------
NuclearInteractionFinder::~NuclearInteractionFinder() {
  delete theNavigationSchool;
}

//----------------------------------------------------------------------
bool  NuclearInteractionFinder::run(const Trajectory& traj, std::auto_ptr<TrajectorySeedCollection>& output) {

        nuclTester->reset();

        if(traj.empty() || !traj.isValid()) return false;

        std::vector<TrajectoryMeasurement> measurements = traj.measurements();

        if(traj.direction()==alongMomentum)  {
                LogDebug("NuclearInteractionFinder") << "NEW TRACK with direction along the momentum\n";
                std::reverse(measurements.begin(), measurements.end());
        }
        else 
            LogDebug("NuclearInteractionFinder") << "NEW TRACK with direction opposite to the momentum\n";

        std::vector<TrajectoryMeasurement>::const_iterator it_meas = measurements.begin();

        std::vector<double> ncompatibleHits;
        bool NIfound = false;

        // Loop on all the RecHits. 
        while(!NIfound)
         {
           if(it_meas == measurements.end()) break;

           nuclTester->push_back(*it_meas, findCompatibleMeasurements(*it_meas, rescaleErrorFactor));
           LogDebug("NuclearInteractionFinder") << "Number of compatible meas:" << (nuclTester->back()).size() << "\n"
                                                << "Mean distance between hits :" << nuclTester->meanHitDistance() << "\n"
                                                << "Mean distance between hits :" << nuclTester->meanEstimate() << "\n";

           // don't check track which reach the end of the tracker
           if( checkCompletedTrack==false && (nuclTester->back()).empty() ) break;

           if(nuclTester->isNuclearInteraction()) NIfound=true;

           ++it_meas;
         }

        if(NIfound) {
            LogDebug("NuclearInteractionFinder") << "NUCLEAR INTERACTION FOUND at index : " << nuclTester->nuclearIndex() << "\n";
            const std::pair<TrajectoryMeasurement, std::vector<TrajectoryMeasurement> >& tmPairs = nuclTester->goodTMPair();

            const TM& innerHit = tmPairs.first;
            const std::vector<TM>& outerHits = tmPairs.second;

            for(std::vector<TM>::const_iterator outhit = outerHits.begin(); outhit!=outerHits.end(); outhit++) {
               if((innerHit.recHit())->isValid() && (outhit->recHit())->isValid()) {
                     theSeed->setMeasurements(innerHit, *outhit);
                     if(theSeed->isValid()) {
                          TrajectorySeed ptraj = theSeed->TrajSeed();
                          output->push_back(ptraj);
                          LogDebug("NuclearInteractionFinder") << "Seed put in event: " << "\n"
                                                           << "Nhits : " << ptraj.nHits() << "\n"
                                                           << "Initial state : " << (ptraj.startingState()).parameters().position() << "\n";
                     }
                     else LogDebug("NuclearInteractionFinder") << "The seed is invalid" << "\n";
                }
                else  LogDebug("NuclearInteractionFinder") << "The initial hits for seeding are invalid" << "\n";
             }
            return true;
        }

    return false;
}
//----------------------------------------------------------------------
std::vector<TrajectoryMeasurement>
NuclearInteractionFinder::findCompatibleMeasurements(const TM& lastMeas, double rescale) const
{
//  double min_pt=1;

  TSOS currentState = lastMeas.updatedState();
  LogDebug("NuclearInteractionFinder") << "currentState :" << currentState << "\n";
/*
  TSOS currentStateError = stateWithLargeError(currentState, min_pt, rescale);
  LogDebug("NuclearInteractionFinder") << "currentStateError :" << currentStateError << "\n";
 //TSOS currentStateErrorInv = stateWithLargeError(currentState, min_pt, -1);
*/
  currentState.rescaleError(rescale);
  return findMeasurementsFromTSOS(currentState, lastMeas);
}

//----------------------------------------------------------------------
std::vector<TrajectoryMeasurement>
NuclearInteractionFinder::findMeasurementsFromTSOS(const TSOS& currentState, const TM& lastMeas) const {

  using namespace std;
  int invalidHits = 0;
  vector<TM> result;
  DetId detid = lastMeas.recHit()->geographicalId();
  const DetLayer* lastLayer = theGeomSearchTracker->detLayer( detid ); //traj.lastLayer();
  vector<const DetLayer*> nl;

  if(lastLayer) { 
          nl = lastLayer->nextLayers( *currentState.freeState(), alongMomentum);
          LogDebug("NuclearInteractionFinder") << "In findCompatibleMeasurements :  number of compatible layers : " << nl.size() << "\n";
  }
  else {
      edm::LogError("NuclearInteractionFinder") << "In findCompatibleMeasurements : lastLayer not accessible";
      return result;
  }

  if (nl.empty()) {
      LogDebug("NuclearInteractionFinder") << "In findCompatibleMeasurements :  no compatible layer found";
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
/*
TrajectoryStateOnSurface NuclearInteractionFinder::stateWithLargeError(const TSOS& state, double min_pt,  int rescale) const {
   // Modification of the momentum = momentum/2
   LocalTrajectoryParameters ltp = state.localParameters();
   AlgebraicVector v = ltp.vector();
   //v[0] = 2*sign*v[0];
   LocalTrajectoryParameters newltp(v, ltp.pzSign(), true);

   // Modification of the error : on the all the parameters by a factor 10
   // on the momentum by a factor 10000  
   AlgebraicSymMatrix m(state.localError().matrix());
   //double sigma = (v[0] > 0) ? fabs(v[0]-1/min_pt) : fabs(v[0]+1/min_pt);
   m*=rescale*rescale;
   //m[0][0] = v[0];

   return TSOS(newltp, m, state.surface(), &(state.globalParameters().magneticField()), state.surfaceSide());
}
*/
