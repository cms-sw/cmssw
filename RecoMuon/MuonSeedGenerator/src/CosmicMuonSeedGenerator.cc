#include "RecoMuon/MuonSeedGenerator/src/CosmicMuonSeedGenerator.h"
/**
 *  CosmicMuonSeedGenerator
 *
 *  $Date$
 *  $Revision$
 *
 *  \author Chang Liu - Purdue University 
 *
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

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
    theDTRecSegmentLabel = pset.getUntrackedParameter<InputTag>("DTRecSegmentLabel");

  if(theEnableCSCFlag)
    theCSCRecSegmentLabel = pset.getUntrackedParameter<InputTag>("CSCRecSegmentLabel");

  // the maximum number of TrajectorySeed
  theMaxSeeds = pset.getParameter<int>("MaxSeeds");

  theMaxDTChi2 = pset.getParameter<double>("MaxDTChi2");
  theMaxCSCChi2 = pset.getParameter<double>("MaxCSCChi2");

}

// Destructor
CosmicMuonSeedGenerator::~CosmicMuonSeedGenerator(){}


// reconstruct muon's seeds
void CosmicMuonSeedGenerator::produce(edm::Event& event, const edm::EventSetup& eSetup){
  
  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  TrajectorySeedCollection seeds;
 
  std::string category = "Muon|RecoMuon|CosmicMuonSeedGenerator";

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
  MuonDetLayerMeasurements muonMeasurements(theDTRecSegmentLabel,theCSCRecSegmentLabel,
					    InputTag(),
					    theEnableDTFlag,theEnableCSCFlag,false);

  muonMeasurements.setEvent(event);

  MuonRecHitContainer allHits;

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

  LogTrace(category)<<"RecHits: Barrel outsideIn "
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

  allHits.insert(allHits.end(),RHMB4.begin(),RHMB4.end());
  allHits.insert(allHits.end(),RHMB3.begin(),RHMB3.end());
  allHits.insert(allHits.end(),RHMB2.begin(),RHMB2.end());
  allHits.insert(allHits.end(),RHMB1.begin(),RHMB1.end());

  stable_sort(allHits.begin(),allHits.end(),DecreasingGlobalY());

  allHits.insert(allHits.end(),RHFME4.begin(),RHFME4.end());
  allHits.insert(allHits.end(),RHFME3.begin(),RHFME3.end());
  allHits.insert(allHits.end(),RHFME2.begin(),RHFME2.end());
  allHits.insert(allHits.end(),RHFME12.begin(),RHFME12.end());
  allHits.insert(allHits.end(),RHFME11.begin(),RHFME11.end());

  allHits.insert(allHits.end(),RHBME4.begin(),RHBME4.end());
  allHits.insert(allHits.end(),RHBME3.begin(),RHBME3.end());
  allHits.insert(allHits.end(),RHBME2.begin(),RHBME2.end());
  allHits.insert(allHits.end(),RHBME12.begin(),RHBME12.end());
  allHits.insert(allHits.end(),RHBME11.begin(),RHBME11.end());

  LogTrace(category)<<"all RecHits: "<<allHits.size();

  if ( !allHits.empty() ) {
    MuonRecHitContainer goodhits = selectSegments(allHits);
    LogTrace(category)<<"good RecHits: "<<goodhits.size();

    if ( goodhits.empty() ) {
      LogTrace(category)<<"No qualified Segments in Event! ";
      LogTrace(category)<<"Use 2D RecHit";

      createSeeds(seeds,allHits,eSetup);

    } 
    else {
      createSeeds(seeds,goodhits,eSetup);
    }

    LogTrace(category)<<"Seeds built: "<<seeds.size();

    for(std::vector<TrajectorySeed>::iterator seed = seeds.begin();
        seed != seeds.end(); ++seed)
        output->push_back(*seed);
    }

  event.put(output);
}


bool CosmicMuonSeedGenerator::checkQuality(const MuonRecHitPointer& hit) const {

  const std::string category = "Muon|RecoMuon|CosmicMuonSeedGenerator";

  // only use 4D segments
  if ( !hit->isValid() ) return false;

  if (hit->dimension() < 4) {
    LogTrace(category)<<"dim < 4";
    return false;
  }

  if (hit->isDT() && ( hit->chi2()> theMaxDTChi2 )) {
    LogTrace(category)<<"DT chi2 too large";
    return false;
  }
  else if (hit->isCSC() &&( hit->chi2()> theMaxCSCChi2 ) ) {
    LogTrace(category)<<"CSC chi2 too large";
     return false;
  }
  return true;

} 

MuonRecHitContainer CosmicMuonSeedGenerator::selectSegments(const MuonRecHitContainer& hits) const {

  MuonRecHitContainer result;
  const std::string category = "Muon|RecoMuon|CosmicMuonSeedGenerator";

  //Only select good quality Segments
  for (MuonRecHitContainer::const_iterator hit = hits.begin(); hit != hits.end(); hit++) {
    if ( checkQuality(*hit) ) result.push_back(*hit);
  }

  if ( result.size() < 2 ) return result;

  MuonRecHitContainer result2;

  //avoid selecting Segments with similar direction
  for (MuonRecHitContainer::iterator hit = result.begin(); hit != result.end(); hit++) {
    if (*hit == 0) continue;
    if ( !(*hit)->isValid() ) continue;
    bool good = true;
    GlobalVector dir1 = (*hit)->globalDirection();
    GlobalPoint pos1 = (*hit)->globalPosition();
    for (MuonRecHitContainer::iterator hit2 = hit + 1; hit2 != result.end(); hit2++) {
        if (*hit2 == 0) continue;
        if ( !(*hit2)->isValid() ) continue;

          //compare direction and position
        GlobalVector dir2 = (*hit2)->globalDirection();
        GlobalPoint pos2 = (*hit2)->globalPosition();
        if ( !areCorrelated((*hit),(*hit2)) ) continue;

        if ( !leftIsBetter((*hit),(*hit2)) ) { 
          good = false;
        } else (*hit2) = 0;
    }

    if ( good ) result2.push_back(*hit);
  }

  result.clear();

  return result2;

}

void CosmicMuonSeedGenerator::createSeeds(TrajectorySeedCollection& results, 
                                          const MuonRecHitContainer& hits, 
                                          const edm::EventSetup& eSetup) const {

  const std::string category = "Muon|RecoMuon|CosmicMuonSeedGenerator";

  if (hits.size() == 0 || results.size() >= theMaxSeeds ) return;
  for (MuonRecHitContainer::const_iterator ihit = hits.begin(); ihit != hits.end(); ihit++) {
    const std::vector<TrajectorySeed>& sds = createSeed((*ihit),eSetup);
    LogTrace(category)<<"created seeds from rechit "<<sds.size();
    results.insert(results.end(),sds.begin(),sds.end());
    if ( results.size() >= theMaxSeeds ) break;
  }
  return;
}

std::vector<TrajectorySeed> CosmicMuonSeedGenerator::createSeed(const MuonRecHitPointer& hit, const edm::EventSetup& eSetup) const {

  std::vector<TrajectorySeed> result;

  const std::string category = "Muon|RecoMuon|CosmicMuonSeedGenerator";

  MuonPatternRecoDumper debug;
  
  edm::ESHandle<MagneticField> field;
  eSetup.get<IdealMagneticFieldRecord>().get(field);

  // set the pt and spt by hand
  double pt = 7.0;
//  double spt = 1.0;
  // FIXME check sign!

  AlgebraicVector t(4);
  AlgebraicSymMatrix mat(5,0) ;

  // Fill the LocalTrajectoryParameters
  LocalPoint segPos=hit->localPosition();
  
  GlobalVector polar(GlobalVector::Spherical(hit->globalDirection().theta(),
                                             hit->globalDirection().phi(),
                                             1.));
  // Force all track downward for cosmic, not beam-halo
  if (hit->globalDirection().eta() < 4.5 && hit->globalDirection().phi() > 0 ) 
    polar = - polar;

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

  LogTrace(category)<<"Trajectory State on Surface of Seed";
  LogTrace(category)<<"mom: "<<tsos.globalMomentum()<<" phi: "<<tsos.globalMomentum().phi();
  LogTrace(category)<<"pos: " << tsos.globalPosition(); 
  LogTrace(category) << "The RecSegment relies on: ";
  LogTrace(category) << debug.dumpMuonId(hit->geographicalId());

  // Transform it in a TrajectoryStateOnSurface
  TrajectoryStateTransform tsTransform;
    
  PTrajectoryStateOnDet *seedTSOS =
    tsTransform.persistentState(tsos ,hit->geographicalId().rawId());
    
  edm::OwnVector<TrackingRecHit> container;
  TrajectorySeed theSeed(*seedTSOS,container,oppositeToMomentum);
  result.push_back(theSeed); 

  return result;
}

bool CosmicMuonSeedGenerator::areCorrelated(const MuonRecHitPointer& lhs, const MuonRecHitPointer& rhs) const {
  bool result = false;

  GlobalVector dir1 = lhs->globalDirection();
  GlobalPoint pos1 = lhs->globalPosition();
  GlobalVector dir2 = rhs->globalDirection();
  GlobalPoint pos2 = rhs->globalPosition();

  GlobalVector dis = pos2 - pos1;

  if ( (deltaEtaPhi(dir1,dir2) < 0.1 || deltaEtaPhi(dir1,-dir2) < 0.1 ) 
        && dis.mag() < 5.0 )
     result = true;

  if ( (deltaEtaPhi(dir1,dir2) < 0.1 ||deltaEtaPhi(dir1,-dir2) < 0.1 ) && 
      (deltaEtaPhi(dir1,dis) < 0.1 || deltaEtaPhi(dir2,dis) < 0.1) ) 
     result = true;

  if ( fabs(dir1.eta()) > 4.0 || fabs(dir2.eta()) > 4.0 ) {
     if ( (fabs(dir1.theta() - dir2.theta()) < 0.07 ||
           fabs(dir1.theta() + dir2.theta()) > 3.07 ) && 
          (fabs(dir1.theta() - dis.theta()) < 0.07 || 
           fabs(dir1.theta() - dis.theta()) < 0.07 ||
           fabs(dir1.theta() + dis.theta()) > 3.07 ||
           fabs(dir1.theta() + dis.theta()) > 3.07 ) )

     result = true;
  }

  return result;
}

float CosmicMuonSeedGenerator::deltaEtaPhi(const GlobalVector& lhs, const GlobalVector& rhs) const {

    float phi1 = lhs.phi();
    float eta1 = lhs.eta();

    float phi2 = rhs.phi();
    float eta2 = rhs.eta();
    float deltaR = sqrt((phi1-phi2)*(phi1-phi2)+(eta1-eta2)*(eta1-eta2));
    return deltaR;

}

bool CosmicMuonSeedGenerator::leftIsBetter(const MuonTransientTrackingRecHit::MuonRecHitPointer& lhs,
                    const MuonTransientTrackingRecHit::MuonRecHitPointer& rhs) const{

     if ( (lhs->degreesOfFreedom() > rhs->degreesOfFreedom() )  ||
          ( (lhs->degreesOfFreedom() == rhs->degreesOfFreedom() ) &&
            (lhs)->chi2() < (rhs)->chi2() ) )  return true;
     else return false;

}


