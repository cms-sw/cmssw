#include "RecoMuon/MuonSeedGenerator/src/CosmicMuonSeedGenerator.h"
/**
 *  CosmicMuonSeedGenerator
 *
 *  $Date: $
 *  $Revision: $
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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

#include <vector>

using namespace std;

// Constructor
CosmicMuonSeedGenerator::CosmicMuonSeedGenerator(const edm::ParameterSet& pset){
  produces<TrajectorySeedCollection>(); 
  
  // the name of the DT rec hits collection
  theDTRecSegmentLabel = pset.getParameter<string>("DTRecSegmentLabel");
  // the name of the CSC rec hits collection
  theCSCRecSegmentLabel = pset.getParameter<string>("CSCRecSegmentLabel");
  // the maximum number of TrajectorySeed
  theMaxSeeds = 100;
  theMaxDTChi2 = 1000.0; //FIXME
  theMaxCSCChi2 = 1000.0;//FIXME

}

// Destructor
CosmicMuonSeedGenerator::~CosmicMuonSeedGenerator(){};


// reconstruct muon's seeds
void CosmicMuonSeedGenerator::produce(edm::Event& event, const edm::EventSetup& eSetup){
  
  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  TrajectorySeedCollection theSeeds;

  // Muon Geometry - DT, CSC and RPC 
  edm::ESHandle<MuonDetLayerGeometry> muonLayers;
  eSetup.get<MuonRecoGeometryRecord>().get(muonLayers);

  // get the DT layers
  vector<DetLayer*> dtLayers = muonLayers->allDTLayers();

  // get the CSC layers
  vector<DetLayer*> cscForwardLayers = muonLayers->forwardCSCLayers();
  vector<DetLayer*> cscBackwardLayers = muonLayers->backwardCSCLayers();
    
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
  MuonDetLayerMeasurements muonMeasurements(true,true,false, 
				    theDTRecSegmentLabel,theCSCRecSegmentLabel);

  muonMeasurements.setEvent(event);

  // ------------        EndCap disk z<0
  RecHitContainer RHBME4 = muonMeasurements.recHits(ME4Bwd);
  RecHitContainer RHBME3 = muonMeasurements.recHits(ME3Bwd);
  RecHitContainer RHBME2 = muonMeasurements.recHits(ME2Bwd);
  RecHitContainer RHBME12 = muonMeasurements.recHits(ME12Bwd);
  RecHitContainer RHBME11 = muonMeasurements.recHits(ME11Bwd);

  // ------------        EndCap disk z>0 
  RecHitContainer RHFME4 = muonMeasurements.recHits(ME4Fwd);
  RecHitContainer RHFME3 = muonMeasurements.recHits(ME3Fwd);
  RecHitContainer RHFME2 = muonMeasurements.recHits(ME2Fwd);
  RecHitContainer RHFME12 = muonMeasurements.recHits(ME12Fwd);
  RecHitContainer RHFME11 = muonMeasurements.recHits(ME11Fwd);

  // ------------        Barrel
  RecHitContainer RHMB4 = muonMeasurements.recHits(MB4DL);
  RecHitContainer RHMB3 = muonMeasurements.recHits(MB3DL);
  RecHitContainer RHMB2 = muonMeasurements.recHits(MB2DL);
  RecHitContainer RHMB1 = muonMeasurements.recHits(MB1DL);

  edm::LogInfo("CosmicMuonSeedGenerator")<<"RecHits: Barrel outsideIn "
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
                                         <<RHFME4.size()<<" : "
                                         <<RHFME3.size()<<" : "
                                         <<RHFME2.size()<<" : "
                                         <<RHFME12.size()<<" : "
                                         <<RHFME11.size()<<" .\n"; 

  // generate Seeds outside in (try the outermost and innermost first)

  createSeeds(theSeeds,RHMB4,eSetup);
  createSeeds(theSeeds,RHFME4,eSetup);
  createSeeds(theSeeds,RHBME4,eSetup);

  createSeeds(theSeeds,RHMB1,eSetup);
  createSeeds(theSeeds,RHFME12,eSetup);
  createSeeds(theSeeds,RHBME12,eSetup);

  createSeeds(theSeeds,RHFME11,eSetup);
  createSeeds(theSeeds,RHBME11,eSetup);

  createSeeds(theSeeds,RHMB3,eSetup);
  createSeeds(theSeeds,RHFME3,eSetup);
  createSeeds(theSeeds,RHBME3,eSetup);


  createSeeds(theSeeds,RHMB2,eSetup);
  createSeeds(theSeeds,RHFME2,eSetup);
  createSeeds(theSeeds,RHBME2,eSetup);

  edm::LogInfo("CosmicMuonSeedGenerator")<<" "<<theSeeds.size()<<". ";


  for(std::vector<TrajectorySeed>::iterator seed = theSeeds.begin();
      seed != theSeeds.end(); ++seed)
      output->push_back(*seed);
  event.put(output);
}


bool CosmicMuonSeedGenerator::checkQuality(MuonTransientTrackingRecHit* hit) const {
  return true; //FIXME

  // only use 4D segments ?  try another way for 2D segments
  if (hit->degreesOfFreedom() < 4) {
    edm::LogInfo("CosmicMuonSeedGenerator")<<"dim < 4";
    return false;
  }
  if (hit->isDT() && ( hit->chi2()> theMaxDTChi2 )) {
    edm::LogInfo("CosmicMuonSeedGenerator")<<"DT chi2 too large";
    return false;
  }
  else if (hit->isCSC() &&( hit->chi2()> theMaxCSCChi2 ) ) {
    edm::LogInfo("CosmicMuonSeedGenerator")<<"CSC chi2 too large";
     return false;
  }
  return true;

} 

void CosmicMuonSeedGenerator::createSeeds(TrajectorySeedCollection& results, 
                                          const RecHitContainer& hits, 
                                          const edm::EventSetup& eSetup) const {

  if (hits.size() == 0 || results.size() >= theMaxSeeds ) return;
  for (RecHitIterator ihit = hits.begin(); ihit != hits.end(); ihit++) {
    if ( !checkQuality(*ihit) ) continue;
    const std::vector<TrajectorySeed>& sds = createSeed((*ihit),eSetup);
    edm::LogInfo("CosmicMuonSeedGenerator")<<"created seeds from rechit "<<sds.size();
    results.insert(results.end(),sds.begin(),sds.end());
  }
  return;
}

std::vector<TrajectorySeed> CosmicMuonSeedGenerator::createSeed(MuonTransientTrackingRecHit* hit, const edm::EventSetup& eSetup) const {

  std::vector<TrajectorySeed> result;

  std::string metname = "CosmicMuonSeedGenerator";

  MuonPatternRecoDumper debug;
  
  edm::ESHandle<MagneticField> field;
  eSetup.get<IdealMagneticFieldRecord>().get(field);

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
  polar *=fabs(pt)/polar.perp();
  LocalVector segDir =hit->det()->toLocal(polar);

  int charge= -1; //(int)(pt/fabs(pt)); //FIXME

  LocalTrajectoryParameters param(segPos,segDir, charge);

  mat = hit->parametersError().similarityT( hit->projectionMatrix() );
  
  float p_err = 0.2; // sqr(spt/(pt*pt)); //FIXME
  mat[0][0]= p_err;
  
  LocalTrajectoryError error(mat);
  
  // Create the TrajectoryStateOnSurface
  TrajectoryStateOnSurface tsos(param, error, hit->det()->surface(), &*field);

  edm::LogInfo(metname)<<"Trajectory State on Surface before the extrapolation";
  edm::LogInfo(metname)<<"mom: "<<tsos.globalMomentum();
  edm::LogInfo(metname)<<" pos: " << tsos.globalPosition(); 

  // Transform it in a TrajectoryStateOnSurface
  TrajectoryStateTransform tsTransform;
     
  PTrajectoryStateOnDet *seedPTSOD =
  tsTransform.persistentState( tsos,hit->geographicalId().rawId());
     
  edm::OwnVector<TrackingRecHit> container;
  TrajectorySeed theSeed(*seedPTSOD,container,alongMomentum);

  result.push_back(theSeed);    
/*
  // set backup seeds with guessed directions 
  // for DT Segment that only hasZed or hasPhi
  // FIXME
*/
  return result;
}
