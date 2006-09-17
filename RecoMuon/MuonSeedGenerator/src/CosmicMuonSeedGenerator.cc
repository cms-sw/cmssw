#include "RecoMuon/MuonSeedGenerator/src/CosmicMuonSeedGenerator.h"
/**
 *  CosmicMuonSeedGenerator
 *
 *  $Date: 2006/09/14 23:51:03 $
 *  $Revision: 1.9 $
 *
 *  \author Chang Liu - Purdue University 
 *
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"


#include <vector>

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

using namespace edm;
using namespace std;

// Constructor
CosmicMuonSeedGenerator::CosmicMuonSeedGenerator(const edm::ParameterSet& pset){
  produces<TrajectorySeedCollection>(); 
  
  // enable the DT chamber
  theEnableDTFlag = pset.getUntrackedParameter<bool>("EnableDTMeasurement",true);
  // enable the CSC chamber
  theEnableCSCFlag = pset.getUntrackedParameter<bool>("EnableCSCMeasurement",true);

  if(theEnableDTFlag)
    theDTRecSegmentLabel = pset.getUntrackedParameter<string>("DTRecSegmentLabel");

  if(theEnableCSCFlag)
    theCSCRecSegmentLabel = pset.getUntrackedParameter<string>("CSCRecSegmentLabel");

  // the maximum number of TrajectorySeed
  theMaxSeeds = pset.getParameter<int>("MaxSeeds");
  theMaxDTChi2 = pset.getParameter<double>("MaxDTChi2");
  theMaxCSCChi2 = pset.getParameter<double>("MaxCSCChi2");

}

// Destructor
CosmicMuonSeedGenerator::~CosmicMuonSeedGenerator(){};


// reconstruct muon's seeds
void CosmicMuonSeedGenerator::produce(edm::Event& event, const edm::EventSetup& eSetup){
  
  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  TrajectorySeedCollection theSeeds;

  // Muon Geometry - DT, CSC and RPC 
  eSetup.get<MuonRecoGeometryRecord>().get(theMuonLayers);

  // get the DT layers
  vector<DetLayer*> dtLayers = theMuonLayers->allDTLayers();

  // get the CSC layers
  vector<DetLayer*> cscForwardLayers = theMuonLayers->forwardCSCLayers();
  vector<DetLayer*> cscBackwardLayers = theMuonLayers->backwardCSCLayers();
    
  // Backward (z<0) EndCap disk
  const DetLayer* ME4Bwd = cscBackwardLayers[4];
  const DetLayer* ME3Bwd = cscBackwardLayers[3];
  const DetLayer* ME2Bwd = cscBackwardLayers[2];
  const DetLayer* ME12Bwd = cscBackwardLayers[1];
  const DetLayer* ME11Bwd = cscBackwardLayers[0];
  
  // Forward (z>0) EndCap disk
  const DetLayer* ME11Fwd = cscForwardLayers[0];
  const DetLayer* ME12Fwd = cscForwardLayers[1];
  const DetLayer* ME2Fwd = cscForwardLayers[2];
  const DetLayer* ME3Fwd = cscForwardLayers[3];
  const DetLayer* ME4Fwd = cscForwardLayers[4];
     
  // barrel
  const DetLayer* MB4DL = dtLayers[3];
  const DetLayer* MB3DL = dtLayers[2];
  const DetLayer* MB2DL = dtLayers[1];
  const DetLayer* MB1DL = dtLayers[0];
  
  // instantiate the accessor
  MuonDetLayerMeasurements muonMeasurements(theEnableDTFlag,theEnableCSCFlag,false, 
				    theDTRecSegmentLabel,theCSCRecSegmentLabel);

  muonMeasurements.setEvent(event);

  // ------------        EndCap disk z<0
  MuonRecHitContainer RHBME4 = muonMeasurements.recHits(ME4Bwd);
  MuonRecHitContainer RHBME3 = muonMeasurements.recHits(ME3Bwd);
  MuonRecHitContainer RHBME2 = muonMeasurements.recHits(ME2Bwd);
  MuonRecHitContainer RHBME12 = muonMeasurements.recHits(ME12Bwd);
  MuonRecHitContainer RHBME11 = muonMeasurements.recHits(ME11Bwd);

  // ------------        EndCap disk z>0 
  MuonRecHitContainer RHFME4 = muonMeasurements.recHits(ME4Fwd);
  MuonRecHitContainer RHFME3 = muonMeasurements.recHits(ME3Fwd);
  MuonRecHitContainer RHFME2 = muonMeasurements.recHits(ME2Fwd);
  MuonRecHitContainer RHFME12 = muonMeasurements.recHits(ME12Fwd);
  MuonRecHitContainer RHFME11 = muonMeasurements.recHits(ME11Fwd);

  // ------------        Barrel
  MuonRecHitContainer RHMB4 = muonMeasurements.recHits(MB4DL);
  MuonRecHitContainer RHMB3 = muonMeasurements.recHits(MB3DL);
  MuonRecHitContainer RHMB2 = muonMeasurements.recHits(MB2DL);
  MuonRecHitContainer RHMB1 = muonMeasurements.recHits(MB1DL);

  LogDebug("CosmicMuonSeedGenerator")<<"RecHits: Barrel outsideIn "
                                     <<RHMB4.size()<<" : "
                                     <<RHMB3.size()<<" : "
                                     <<RHMB2.size()<<" : "
                                     <<RHMB1.size()<<" .\n"
                                     <<"RecHits: Forward Endcap outsideIn "
                                     <<RHFME4.size()<<" : "
                                     <<RHFME3.size()<<" : "
                                     <<RHFME2.size()<<" : "
                                     <<RHFME12.size()<<" : "
                                     <<RHFME11.size()<<" .\n"
                                     <<"RecHits: Backward Endcap outsideIn "
                                     <<RHBME4.size()<<" : "
                                     <<RHBME3.size()<<" : "
                                     <<RHBME2.size()<<" : "
                                     <<RHBME12.size()<<" : "
                                     <<RHBME11.size()<<" .\n"; 

  // generate Seeds upside-down (try the nnermost first)
  // only lower part works now

  createSeeds(theSeeds,RHMB1,eSetup);
  createSeeds(theSeeds,RHFME12,eSetup);
  createSeeds(theSeeds,RHBME12,eSetup);

  createSeeds(theSeeds,RHFME11,eSetup);
  createSeeds(theSeeds,RHBME11,eSetup);

  createSeeds(theSeeds,RHFME2,eSetup);
  createSeeds(theSeeds,RHBME2,eSetup);

  createSeeds(theSeeds,RHFME3,eSetup);
  createSeeds(theSeeds,RHBME3,eSetup);

  if ( theSeeds.empty() ) {

    createSeeds(theSeeds,RHMB2,eSetup);

    createSeeds(theSeeds,RHMB3,eSetup);

    createSeeds(theSeeds,RHMB4,eSetup);
    createSeeds(theSeeds,RHFME4,eSetup);
    createSeeds(theSeeds,RHBME4,eSetup);

  }

  for(std::vector<TrajectorySeed>::iterator seed = theSeeds.begin();
      seed != theSeeds.end(); ++seed)
      output->push_back(*seed);
  event.put(output);
}


bool CosmicMuonSeedGenerator::checkQuality(MuonRecHitPointer hit) const {

  // only use 4D segments ?  try another way for 2D segments
  if (hit->degreesOfFreedom() < 4) {
    LogDebug("CosmicMuonSeedGenerator")<<"dim < 4";
    return false;
  }
  if (hit->isDT() && ( hit->chi2()> theMaxDTChi2 )) {
    LogDebug("CosmicMuonSeedGenerator")<<"DT chi2 too large";
    return false;
  }
  else if (hit->isCSC() &&( hit->chi2()> theMaxCSCChi2 ) ) {
    LogDebug("CosmicMuonSeedGenerator")<<"CSC chi2 too large";
     return false;
  }
  return true;

} 

void CosmicMuonSeedGenerator::createSeeds(TrajectorySeedCollection& results, 
                                          const MuonRecHitContainer& hits, 
                                          const edm::EventSetup& eSetup) const {

  if (hits.size() == 0 || results.size() >= theMaxSeeds ) return;
  for (MuonRecHitContainer::const_iterator ihit = hits.begin(); ihit != hits.end(); ihit++) {
    if ( !checkQuality(*ihit)) continue;
    const std::vector<TrajectorySeed>& sds = createSeed((*ihit),eSetup);
    LogDebug("CosmicMuonSeedGenerator")<<"created seeds from rechit "<<sds.size();
    results.insert(results.end(),sds.begin(),sds.end());
    if ( results.size() >= theMaxSeeds ) break;
  }
  return;
}

std::vector<TrajectorySeed> CosmicMuonSeedGenerator::createSeed(MuonRecHitPointer hit, const edm::EventSetup& eSetup) const {

  std::vector<TrajectorySeed> result;

  const std::string metname = "Muon|RecoMuon|CosmicMuonSeedGenerator";

  MuonPatternRecoDumper debug;
  
  edm::ESHandle<MagneticField> field;
  eSetup.get<IdealMagneticFieldRecord>().get(field);
  // FIXME: put it into a parameter set  
  edm::ESHandle<Chi2MeasurementEstimatorBase> estimator;
  eSetup.get<TrackingComponentsRecord>().get("Chi2MeasurementEstimator",estimator);
  
  // FIXME: put it into a parameter set
  edm::ESHandle<Propagator> propagator;
  eSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny",propagator);


  // set the pt and spt by hand
  double pt = 5.0;
//  double spt = 1.0;
  // FIXME check sign!

  AlgebraicVector t(4);
  AlgebraicSymMatrix mat(5,0) ;

  // Fill the LocalTrajectoryParameters
  LocalPoint segPos=hit->localPosition();

  
  GlobalVector polar(GlobalVector::Spherical(hit->globalDirection().theta(),
                                             hit->globalDirection().phi(),
                                             1.));
  // Force all track downward

  if (hit->globalDirection().phi() > 0 )  polar = - polar;

  polar *=fabs(pt)/polar.perp();

  LocalVector segDir =hit->det()->toLocal(polar);

  int charge= 1; //more mu+ than mu- in natural  //(int)(pt/fabs(pt)); //FIXME

  LocalTrajectoryParameters param(segPos,segDir, charge);

  mat = hit->parametersError().similarityT( hit->projectionMatrix() );
  
  float p_err = 0.2; // sqr(spt/(pt*pt)); //FIXME
  mat[0][0]= p_err;
  
  LocalTrajectoryError error(mat);
  
  // Create the TrajectoryStateOnSurface
  TrajectoryStateOnSurface tsos(param, error, hit->det()->surface(), &*field);

  LogDebug(metname)<<"Trajectory State on Surface before the extrapolation";
  LogDebug(metname)<<"mom: "<<tsos.globalMomentum()<<" phi: "<<tsos.globalMomentum().phi();
  LogDebug(metname)<<"pos: " << tsos.globalPosition(); 
  LogDebug(metname) << "The RecSegment relies on: "<<endl;
  LogDebug(metname) << debug.dumpMuonId(hit->geographicalId());

 // ask for compatible layers
  DirectMuonNavigation theNavigation(&*theMuonLayers);
  vector<const DetLayer*> detLayers = theNavigation.compatibleLayers(*(tsos.freeState()),oppositeToMomentum);
  
  LogDebug(metname) << "There are "<< detLayers.size() <<" compatible layers"<<endl;
  
  vector<DetWithState> detsWithStates;

  if(detLayers.size()){
    LogDebug(metname) <<"Compatible layers:"<<endl;
    for( vector<const DetLayer*>::const_iterator layer = detLayers.begin(); 
	 layer != detLayers.end(); layer++){
      LogDebug(metname) << debug.dumpMuonId((*layer)->basicComponents().front()->geographicalId());
      LogDebug(metname) << debug.dumpLayer(*layer);
    }
    
    // ask for compatible dets
    // ask for compatible dets
    LogDebug(metname) <<"first one:"<<endl;
    LogDebug(metname) << debug.dumpLayer(detLayers.front());

    detsWithStates = detLayers.front()->compatibleDets(tsos, *propagator, *estimator);
    LogDebug(metname)<<"Number of compatible dets: "<<detsWithStates.size()<<endl;
  }
  else
    LogDebug(metname)<<"No compatible layers found"<<endl;

  if(detsWithStates.size()){
    // get the updated TSOS
    TrajectoryStateOnSurface newTSOS = detsWithStates.front().second;
    const GeomDet *newTSOSDet = detsWithStates.front().first;
    
    if ( newTSOS.isValid() ) {
      
      LogDebug(metname)<<"New TSOS is on det: "<<endl;
      LogDebug(metname) << debug.dumpMuonId(newTSOSDet->geographicalId());

      LogDebug(metname) << "Trajectory State on Surface after the extrapolation"<<endl;
      LogDebug(metname)<<"mom: "<<newTSOS.globalMomentum();
      LogDebug(metname)<<"pos: " << newTSOS.globalPosition();


      // Transform it in a TrajectoryStateOnSurface
      TrajectoryStateTransform tsTransform;
      
      PTrajectoryStateOnDet *seedTSOS =
	tsTransform.persistentState( newTSOS,newTSOSDet->geographicalId().rawId());
      
      edm::OwnVector<TrackingRecHit> container;
      TrajectorySeed theSeed(*seedTSOS,container,oppositeToMomentum);
      
      result.push_back(theSeed);
    } 
  }
  else{
    
    // Transform it in a TrajectoryStateOnSurface
    TrajectoryStateTransform tsTransform;
    
    PTrajectoryStateOnDet *seedTSOS =
      tsTransform.persistentState(tsos ,hit->geographicalId().rawId());
    
    edm::OwnVector<TrackingRecHit> container;
    TrajectorySeed theSeed(*seedTSOS,container,oppositeToMomentum);
    result.push_back(theSeed); 
  }

  return result;
}
