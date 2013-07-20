#include "RecoMuon/MuonSeedGenerator/plugins/CosmicMuonSeedGenerator.h"
/**
 *  CosmicMuonSeedGenerator
 *
 *  $Date: 2011/12/23 05:05:35 $
 *  $Revision: 1.8 $
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
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <vector>

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

using namespace edm;
using namespace std;

// Constructor
CosmicMuonSeedGenerator::CosmicMuonSeedGenerator(const edm::ParameterSet& pset){
  produces<TrajectorySeedCollection>(); 
  
  // enable the DT chamber
  theEnableDTFlag = pset.getParameter<bool>("EnableDTMeasurement");
  // enable the CSC chamber
  theEnableCSCFlag = pset.getParameter<bool>("EnableCSCMeasurement");

  theDTRecSegmentLabel = pset.getParameter<InputTag>("DTRecSegmentLabel");

  theCSCRecSegmentLabel = pset.getParameter<InputTag>("CSCRecSegmentLabel");

  // the maximum number of TrajectorySeed
  theMaxSeeds = pset.getParameter<int>("MaxSeeds");

  theMaxDTChi2 = pset.getParameter<double>("MaxDTChi2");
  theMaxCSCChi2 = pset.getParameter<double>("MaxCSCChi2");

  // pre-determined parameters for seed pt calculation ( pt * dphi )
  theParameters["topmb41"] = 0.87;
  theParameters["bottommb41"] = 1.2;
  theParameters["topmb42"] = 0.67;
  theParameters["bottommb42"] = 0.98;
  theParameters["topmb43"] = 0.34;
  theParameters["bottommb43"] = 0.58;
  theParameters["topmb31"] = 0.54;
  theParameters["bottommb31"] = 0.77;
  theParameters["topmb32"] = 0.35;
  theParameters["bottommb32"] = 0.55;
  theParameters["topmb21"] = 0.21;
  theParameters["bottommb21"] = 0.31;

}

// Destructor
CosmicMuonSeedGenerator::~CosmicMuonSeedGenerator(){
}


// reconstruct muon's seeds
void CosmicMuonSeedGenerator::produce(edm::Event& event, const edm::EventSetup& eSetup){

  eSetup.get<IdealMagneticFieldRecord>().get(theField);

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
     
  MuonDetLayerMeasurements muonMeasurements(theDTRecSegmentLabel,theCSCRecSegmentLabel,
					    InputTag(),
					    theEnableDTFlag,theEnableCSCFlag,false);

  muonMeasurements.setEvent(event);

  MuonRecHitContainer allHits;

  vector<MuonRecHitContainer> RHMBs;
  vector<MuonRecHitContainer> RHMEFs;
  vector<MuonRecHitContainer> RHMEBs;

  stable_sort(allHits.begin(),allHits.end(),DecreasingGlobalY());

  for (vector<DetLayer*>::reverse_iterator icsclayer = cscForwardLayers.rbegin();
       icsclayer != cscForwardLayers.rend() - 1; ++icsclayer) {
       
       MuonRecHitContainer RHMF = muonMeasurements.recHits(*icsclayer);
       allHits.insert(allHits.end(),RHMF.begin(),RHMF.end());

  }

  for (vector<DetLayer*>::reverse_iterator icsclayer = cscBackwardLayers.rbegin();
       icsclayer != cscBackwardLayers.rend() - 1; ++icsclayer) {

       MuonRecHitContainer RHMF = muonMeasurements.recHits(*icsclayer);
       allHits.insert(allHits.end(),RHMF.begin(),RHMF.end());

  }

  for (vector<DetLayer*>::reverse_iterator idtlayer = dtLayers.rbegin();
       idtlayer != dtLayers.rend(); ++idtlayer) {

       MuonRecHitContainer RHMB = muonMeasurements.recHits(*idtlayer);
       RHMBs.push_back(RHMB);

       if ( idtlayer != dtLayers.rbegin() ) allHits.insert(allHits.end(),RHMB.begin(),RHMB.end());

  }

//  stable_sort(allHits.begin(),allHits.end(),DecreasingGlobalY());

  LogTrace(category)<<"all RecHits: "<<allHits.size();

//  CosmicMuonSeedGenerator::MuonRecHitPairVector mb41 = makeSegPairs(RHMBs[0], RHMBs[3], "mb41");
//  createSeeds(seeds, mb41, eSetup);

//  CosmicMuonSeedGenerator::MuonRecHitPairVector mb43 = makeSegPairs(RHMBs[0],RHMBs[1], "mb43");
//  createSeeds(seeds, mb43, eSetup);

  CosmicMuonSeedGenerator::MuonRecHitPairVector mb42 = makeSegPairs(RHMBs[0],RHMBs[2], "mb42");
  createSeeds(seeds, mb42, eSetup);

//  CosmicMuonSeedGenerator::MuonRecHitPairVector mb32 = makeSegPairs(RHMBs[1], RHMBs[2], "mb32");
//  createSeeds(seeds, mb32, eSetup);

  CosmicMuonSeedGenerator::MuonRecHitPairVector mb31 = makeSegPairs(RHMBs[1], RHMBs[3], "mb31");
  createSeeds(seeds, mb31, eSetup);

//  CosmicMuonSeedGenerator::MuonRecHitPairVector mb21 = makeSegPairs(RHMBs[2], RHMBs[3], "mb21");
//  createSeeds(seeds, mb21, eSetup);

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
  }

  LogTrace(category)<<"Seeds built: "<<seeds.size();

  for(std::vector<TrajectorySeed>::iterator seed = seeds.begin();
      seed != seeds.end(); ++seed) {
        output->push_back(*seed);
  }

  event.put(output);
  seeds.clear();

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
    //UNUSED:    GlobalVector dir1 = (*hit)->globalDirection();
    //UNUSED:    GlobalPoint pos1 = (*hit)->globalPosition();
    for (MuonRecHitContainer::iterator hit2 = hit + 1; hit2 != result.end(); hit2++) {
        if (*hit2 == 0) continue;
        if ( !(*hit2)->isValid() ) continue;

          //compare direction and position
	//UNUSED:        GlobalVector dir2 = (*hit2)->globalDirection();
	//UNUSED:        GlobalPoint pos2 = (*hit2)->globalPosition();
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

void CosmicMuonSeedGenerator::createSeeds(TrajectorySeedCollection& results,
                                          const CosmicMuonSeedGenerator::MuonRecHitPairVector& hitpairs,
                                          const edm::EventSetup& eSetup) const {

  const std::string category = "Muon|RecoMuon|CosmicMuonSeedGenerator";

  if (hitpairs.size() == 0 || results.size() >= theMaxSeeds ) return;
  for (CosmicMuonSeedGenerator::MuonRecHitPairVector::const_iterator ihitpair = hitpairs.begin(); ihitpair != hitpairs.end(); ihitpair++) {
    const std::vector<TrajectorySeed>& sds = createSeed((*ihitpair),eSetup);
    LogTrace(category)<<"created seeds from rechit "<<sds.size();
    results.insert(results.end(),sds.begin(),sds.end());
    if ( results.size() >= theMaxSeeds ) break;
  }
  return;
}


std::vector<TrajectorySeed> CosmicMuonSeedGenerator::createSeed(const MuonRecHitPointer& hit, const edm::EventSetup& eSetup) const {

  std::vector<TrajectorySeed> result;

  const std::string category = "Muon|RecoMuon|CosmicMuonSeedGenerator";

  MuonPatternRecoDumper dumper;
  
  // set the pt by hand
  double pt = 10.0;

  // AlgebraicVector4 t;
  AlgebraicSymMatrix mat(5,0);

  // Fill the LocalTrajectoryParameters
  LocalPoint segPos=hit->localPosition();
  
  GlobalVector polar(GlobalVector::Spherical(hit->globalDirection().theta(),
                                             hit->globalDirection().phi(),
                                             1.));
  // Force all track downward for cosmic, not beam-halo
  if (hit->geographicalId().subdetId() == MuonSubdetId::DT && fabs(hit->globalDirection().eta()) < 4.0 && hit->globalDirection().phi() > 0 ) 
    polar = - polar;

  if (hit->geographicalId().subdetId() == MuonSubdetId::CSC && fabs(hit->globalDirection().eta()) > 2.3 ) {
    polar = - polar;
  }

  polar *=fabs(pt)/polar.perp();

  LocalVector segDir =hit->det()->toLocal(polar);

  int charge= 1;
  LocalTrajectoryParameters param(segPos,segDir, charge);

  charge= -1;
  LocalTrajectoryParameters param2(segPos,segDir, charge);

  mat = hit->parametersError().similarityT( hit->projectionMatrix() );
  
  float p_err = 0.2;
  mat[0][0]= p_err;
  
  LocalTrajectoryError error(asSMatrix<5>(mat));
  
  // Create the TrajectoryStateOnSurface
  TrajectoryStateOnSurface tsos(param, error, hit->det()->surface(), &*theField);
  TrajectoryStateOnSurface tsos2(param2, error, hit->det()->surface(), &*theField);

  LogTrace(category)<<"Trajectory State on Surface of Seed";
  LogTrace(category)<<"mom: "<<tsos.globalMomentum()<<" phi: "<<tsos.globalMomentum().phi();
  LogTrace(category)<<"pos: " << tsos.globalPosition(); 
  LogTrace(category) << "The RecSegment relies on: ";
  LogTrace(category) << dumper.dumpMuonId(hit->geographicalId());

  result.push_back( tsosToSeed(tsos, hit->geographicalId().rawId()) ); 
  result.push_back( tsosToSeed(tsos2, hit->geographicalId().rawId()) );

  return result;
}

bool CosmicMuonSeedGenerator::areCorrelated(const MuonRecHitPointer& lhs, const MuonRecHitPointer& rhs) const {
  bool result = false;

  GlobalVector dir1 = lhs->globalDirection();
  GlobalPoint pos1 = lhs->globalPosition();
  GlobalVector dir2 = rhs->globalDirection();
  GlobalPoint pos2 = rhs->globalPosition();

  GlobalVector dis = pos2 - pos1;

  if ( (deltaR<double>(dir1.eta(), dir1.phi(), dir2.eta(), dir2.phi()) < 0.1 || deltaR<double>(dir1.eta(), dir1.phi(), -dir2.eta(), -dir2.phi()) < 0.1 ) 
        && dis.mag() < 5.0 )
     result = true;

  if ( (deltaR<double>(dir1.eta(), dir1.phi(), dir2.eta(), dir2.phi()) < 0.1 || deltaR<double>(dir1.eta(), dir1.phi(), -dir2.eta(), -dir2.phi()) < 0.1 ) &&
       (deltaR<double>(dir1.eta(), dir1.phi(), dis.eta(), dis.phi()) < 0.1 || deltaR<double>(dir2.eta(), dir2.phi(), dis.eta(), dis.phi()) < 0.1 ) )
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

bool CosmicMuonSeedGenerator::leftIsBetter(const MuonTransientTrackingRecHit::MuonRecHitPointer& lhs,
                    const MuonTransientTrackingRecHit::MuonRecHitPointer& rhs) const{

     if ( (lhs->degreesOfFreedom() > rhs->degreesOfFreedom() )  ||
          ( (lhs->degreesOfFreedom() == rhs->degreesOfFreedom() ) &&
            (lhs)->chi2() < (rhs)->chi2() ) )  return true;
     else return false;

}


CosmicMuonSeedGenerator::MuonRecHitPairVector
CosmicMuonSeedGenerator::makeSegPairs(const MuonTransientTrackingRecHit::MuonRecHitContainer& hits1, const MuonTransientTrackingRecHit::MuonRecHitContainer& hits2, std::string tag) const {

   MuonRecHitPairVector result;
  const std::string category = "Muon|RecoMuon|CosmicMuonSeedGenerator";

   if (hits1.empty() || hits2.empty()  )  return result;

   for (MuonRecHitContainer::const_iterator ihit1 = hits1.begin(); ihit1 != hits1.end(); ihit1++) {
     if ( !checkQuality(*ihit1) ) continue;

     for (MuonRecHitContainer::const_iterator ihit2 = hits2.begin(); ihit2 != hits2.end(); ihit2++) {
        if ( !checkQuality(*ihit2) ) continue;

        float dphi = deltaPhi((*ihit1)->globalPosition().phi(), (*ihit2)->globalPosition().phi());
        if ( dphi < 0.5 ) {
	   if ((*ihit1)->globalPosition().y() > 0.0 && ( (*ihit1)->globalPosition().y()  > (*ihit2)->globalPosition().y() ) ) { 
              std::string tag2 = "top"+tag;

              result.push_back(MuonRecHitPair(*ihit1, *ihit2, tag2));
           } else if ((*ihit1)->globalPosition().y() < 0.0 && ( (*ihit1)->globalPosition().y()  < (*ihit2)->globalPosition().y() ) ) {
              std::string tag2 = "bottom"+tag;
              result.push_back(MuonRecHitPair(*ihit2, *ihit1, tag2));

           }
        }
     }
   }

   return result;
}

std::vector<TrajectorySeed> CosmicMuonSeedGenerator::createSeed(const CosmicMuonSeedGenerator::MuonRecHitPair& hitpair,
                                         const edm::EventSetup& eSetup) const {
   std::vector<TrajectorySeed> result; 

  const std::string category = "Muon|RecoMuon|CosmicMuonSeedGenerator";

  MuonPatternRecoDumper dumper;
  
  float dphi = deltaPhi((hitpair.first)->globalDirection().phi(), (hitpair.second)->globalDirection().phi());

  LogTrace(category)<<"hitpair.type "<<hitpair.type; 

  map<string, float>::const_iterator iterPar = theParameters.find(hitpair.type);
  if ( iterPar == theParameters.end() ) {
       return result;
  }

  // set the pt and charge by dphi
  int charge = (dphi > 0) ? -1 : 1;

  double pt = 999.0;
  float paraC = (iterPar->second);

  if (fabs(dphi) > 1e-5) {
    pt = paraC/fabs(dphi); 
  }

  if (pt < 10.0 ) { return result; } //still use the old strategy for low pt

  AlgebraicVector t(4);
  AlgebraicSymMatrix mat(5,0) ;

  MuonTransientTrackingRecHit::MuonRecHitPointer hit = hitpair.first;
  if ( hit->dimension() < (hitpair.second)->dimension() ) hit = hitpair.second;

  // Fill the LocalTrajectoryParameters
  LocalPoint segPos=hit->localPosition();
  
  GlobalVector polar(GlobalVector::Spherical(hit->globalDirection().theta(),
                                             hit->globalDirection().phi(),
                                             1.));
  // Force all track downward for cosmic, not beam-halo
  if (hit->geographicalId().subdetId() == MuonSubdetId::DT && fabs(hit->globalDirection().eta()) < 4.0 && hit->globalDirection().phi() > 0 ) 
    polar = - polar;

  if (hit->geographicalId().subdetId() == MuonSubdetId::CSC && fabs(hit->globalDirection().eta()) > 2.3 ) {
    polar = - polar;
  }

  polar *=fabs(pt)/polar.perp();

  LocalVector segDir =hit->det()->toLocal(polar);

  LocalTrajectoryParameters param(segPos,segDir, charge);

  mat = hit->parametersError().similarityT( hit->projectionMatrix() );
  
  float p_err = 0.004/paraC;
  if (pt < 10.01) p_err = 0.1; 
  mat[0][0]= p_err;
  
  LocalTrajectoryError error(asSMatrix<5>(mat));
  
  // Create the TrajectoryStateOnSurface
  TrajectoryStateOnSurface tsos(param, error, hit->det()->surface(), &*theField);

  LogTrace(category)<<"Trajectory State on Surface of Seed";
  LogTrace(category)<<"mom: "<<tsos.globalMomentum()<<" phi: "<<tsos.globalMomentum().phi();
  LogTrace(category)<<"pos: " << tsos.globalPosition(); 
  LogTrace(category) << "The RecSegment relies on: ";
  LogTrace(category) << dumper.dumpMuonId(hit->geographicalId());

  result.push_back( tsosToSeed(tsos, hit->geographicalId().rawId()) );

   return result;
}

TrajectorySeed CosmicMuonSeedGenerator::tsosToSeed(const TrajectoryStateOnSurface& tsos, uint32_t id) const {

  PTrajectoryStateOnDet const & seedTSOS = trajectoryStateTransform::persistentState(tsos, id);

  edm::OwnVector<TrackingRecHit> container;
  TrajectorySeed seed(seedTSOS,container,alongMomentum);
  return seed;
}

