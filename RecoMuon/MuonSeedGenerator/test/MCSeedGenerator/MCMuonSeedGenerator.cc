// Class Header
#include "RecoMuon/MuonSeedGenerator/test/MCSeedGenerator/MCMuonSeedGenerator.h"

// Data Formats 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/Common/interface/Handle.h"
 
// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"



DEFINE_FWK_MODULE(MCMuonSeedGenerator2);

using namespace std;
using namespace edm;

// constructors
MCMuonSeedGenerator2::MCMuonSeedGenerator2(const edm::ParameterSet& parameterSet):theSeedType(FromTracks){ 
  
  theCSCSimHitLabel = parameterSet.getParameter<InputTag>("CSCSimHit");
  theDTSimHitLabel = parameterSet.getParameter<InputTag>("DTSimHit");
  theRPCSimHitLabel = parameterSet.getParameter<InputTag>("RPCSimHit");
  theSimTrackLabel = parameterSet.getParameter<InputTag>("SimTrack");
  theSimVertexLabel = parameterSet.getParameter<InputTag>("SimVertex");

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);

  // Error Scale
  theErrorScale =  parameterSet.getParameter<double>("ErrorScale");

  string seedType = parameterSet.getParameter<string>("SeedType");

  if(seedType == "FromHits") theSeedType = FromHits;
  else if(seedType == "FromTracks") theSeedType = FromTracks;
  else LogError("Muon|RecoMuon|MCMuonSeedGenerator2") << "Wrong seed type!!! "<<seedType;
  
  produces<TrajectorySeedCollection>(); 
}

// destructor
MCMuonSeedGenerator2::~MCMuonSeedGenerator2(){
  if (theService) delete theService;
}

void MCMuonSeedGenerator2::produce(edm::Event& event, const edm::EventSetup& setup)
{
  const std::string metname = "Muon|RecoMuon|MCMuonSeedGenerator2";

  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  // Update the services
  theService->update(setup);
  
  // Get the SimHit collection from the event
  Handle<PSimHitContainer> dtSimHits;
  event.getByLabel(theDTSimHitLabel.instance(),theDTSimHitLabel.label(), dtSimHits);
  
  Handle<PSimHitContainer> cscSimHits;
  event.getByLabel(theCSCSimHitLabel.instance(),theCSCSimHitLabel.label(), cscSimHits);
  
  Handle<PSimHitContainer> rpcSimHits;
  event.getByLabel(theRPCSimHitLabel.instance(),theRPCSimHitLabel.label(), rpcSimHits);  

  Handle<SimTrackContainer> simTracks;
  event.getByLabel(theSimTrackLabel.label(),simTracks);

  Handle<SimVertexContainer> simVertices;
  event.getByLabel<SimVertexContainer>(theSimVertexLabel, simVertices);



  map<unsigned int, vector<const PSimHit*> > mapOfMuonSimHits;
  
  for(PSimHitContainer::const_iterator simhit = dtSimHits->begin();
      simhit != dtSimHits->end(); ++simhit) {
    if (abs(simhit->particleType()) != 13) continue;
    mapOfMuonSimHits[simhit->trackId()].push_back(&*simhit);
  }

  for(PSimHitContainer::const_iterator simhit = cscSimHits->begin();
      simhit != cscSimHits->end(); ++simhit) {
    if (abs(simhit->particleType()) != 13) continue;
    mapOfMuonSimHits[simhit->trackId()].push_back(&*simhit);
  }

  for(PSimHitContainer::const_iterator simhit = rpcSimHits->begin();
      simhit != rpcSimHits->end(); ++simhit) {
    if (abs(simhit->particleType()) != 13) continue;
    mapOfMuonSimHits[simhit->trackId()].push_back(&*simhit);
  }


  for (SimTrackContainer::const_iterator simTrack = simTracks->begin(); 
       simTrack != simTracks->end(); ++simTrack){

    if (abs(simTrack->type()) != 13) continue;
    
    map<unsigned int, vector<const PSimHit*> >::const_iterator mapIterator = 
      mapOfMuonSimHits.find(simTrack->trackId());
    
    if (mapIterator == mapOfMuonSimHits.end() ){
      LogTrace(metname)<<"...Very strange, no sim hits associated to the sim track!"<<"\n"
		       <<"SimTrack's eta: "<<simTrack->momentum().eta();
      continue;
    }
    
    vector<const PSimHit*> muonSimHits = mapIterator->second;

    if(muonSimHits.size() < 1) continue;
    
    stable_sort(muonSimHits.begin(),muonSimHits.end(),RadiusComparatorInOut(theService->trackingGeometry()));
    
    const PSimHit* innerSimHit = muonSimHits.front();

    TrajectorySeed* seed = 0;

    // switch(theSeedType){

//     case FromHits:{
//       seed = createSeedFromHit(innerSimHit);
//       break;}
      
//     case FromTracks:{
//       seed = createSeedFromTrack(*simTrack, (*simVertices)[simTrack->vertIndex()], DetId(innerSimHit->detUnitId()));
//       break;}
//     }
    
    if(theSeedType == FromTracks) 
      seed = createSeedFromTrack(*simTrack, (*simVertices)[simTrack->vertIndex()], DetId(innerSimHit->detUnitId()));
    else if(theSeedType == FromHits) 
      seed = createSeedFromHit(innerSimHit);
    else{
      LogError(metname)<<"ERROR!!";
      continue;
    }

    if(seed) output->push_back(*seed);
  }

  event.put(output);
}



TrajectorySeed* MCMuonSeedGenerator2::createSeedFromHit(const PSimHit* innerSimHit){
  
  const std::string metname = "Muon|RecoMuon|MCMuonSeedGenerator2";
  MuonPatternRecoDumper debug;


  LogTrace(metname) << "Seed from sim hit";

  const GeomDet *geomDet = theService->trackingGeometry()->idToDet(DetId(innerSimHit->detUnitId()));  
  LogTrace(metname) << "Seed geom det: "<<debug.dumpMuonId(geomDet->geographicalId());

  GlobalPoint glbPosition = geomDet->toGlobal(innerSimHit->localPosition());
  GlobalVector glbMomentum = geomDet->toGlobal(innerSimHit->momentumAtEntry());
  
  const GlobalTrajectoryParameters globalParameters(glbPosition,glbMomentum,
						    -innerSimHit->particleType()/ abs(innerSimHit->particleType()),
						    &*theService->magneticField());
  
  AlgebraicSymMatrix66 covarianceMatrix = AlgebraicMatrixID();
  covarianceMatrix *= theErrorScale;
  
  const CartesianTrajectoryError cartesianErrors(covarianceMatrix);

  
  TrajectoryStateOnSurface tsos(globalParameters,
				cartesianErrors,
				geomDet->surface());
    
  LogTrace(metname) << "State on: "<<debug.dumpMuonId(DetId(innerSimHit->detUnitId()));
  LogTrace(metname) << debug.dumpTSOS(tsos);
  
  // convert the TSOS into a PTSOD
  PTrajectoryStateOnDet seedTSOS = trajectoryStateTransform::persistentState(tsos,geomDet->geographicalId().rawId());
  
  edm::OwnVector<TrackingRecHit> container;
  TrajectorySeed* seed = new TrajectorySeed(seedTSOS,container,alongMomentum);

  return seed;  
}



TrajectorySeed* MCMuonSeedGenerator2::createSeedFromTrack(const SimTrack &simTrack, const SimVertex &simVertex, DetId detId){
  
  const std::string metname = "Muon|RecoMuon|MCMuonSeedGenerator2";
  MuonPatternRecoDumper debug;

  LogTrace(metname) << "Seed from sim track";

  TrajectorySeed* seed = 0;

  const GeomDet *geomDet = theService->trackingGeometry()->idToDet(detId);  
  LogTrace(metname) << "Seed geom det: "<<debug.dumpMuonId(geomDet->geographicalId());

    
  GlobalVector simMomentum(simTrack.momentum().x(),simTrack.momentum().y(),simTrack.momentum().z());
  
  GlobalPoint simPosition(0.,0.,0.);
  if (simTrack.vertIndex() >= 0)
    simPosition = GlobalPoint(simVertex.position().x(),
			      simVertex.position().y(),
			      simVertex.position().z());
  
  AlgebraicSymMatrix66 matI = AlgebraicMatrixID(); 
  matI *= 100; // 1e-20; 
  CartesianTrajectoryError simCov(matI);

  GlobalTrajectoryParameters parameters( simPosition, simMomentum, int(simTrack.charge()), &*theService->magneticField());
  FreeTrajectoryState simFTS(parameters, simCov);

  LogTrace(metname) << "FTS from the Seed";
  LogTrace(metname) << debug.dumpFTS(simFTS);

  TrajectoryStateOnSurface simSeedTSOS =  theService->propagator("SteppingHelixPropagatorAlong")->propagate(simFTS,geomDet->surface());
  
  if(!simSeedTSOS.isValid()){
    LogTrace(metname)<<"Propagation from IP to the hit layer failed";
    return seed;
  }

  // convert the TSOS into a PTSOD
  PTrajectoryStateOnDet const & seedTSOS = trajectoryStateTransform::persistentState(simSeedTSOS,geomDet->geographicalId().rawId());

  LogTrace(metname) << "State on: "<<debug.dumpMuonId(detId);
  LogTrace(metname) << debug.dumpTSOS(simSeedTSOS);

  edm::OwnVector<TrackingRecHit> container;
  seed = new TrajectorySeed(seedTSOS,container,alongMomentum);
  
  return seed;  
}



