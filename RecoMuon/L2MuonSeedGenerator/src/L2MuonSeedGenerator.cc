//-------------------------------------------------
//
/**  \class L2MuonSeedGenerator
 * 
 *   L2 muon seed generator:
 *   Transform the L1 informations in seeds for the
 *   L2 muon reconstruction
 *
 *
 *   $Date: 2006/11/13 13:49:44 $
 *   $Revision: 1.4 $
 *
 *   \author  A.Everett, R.Bellan
 *
 *    ORCA's author: N. Neumeister 
 */
//
//--------------------------------------------------

// Class Header
#include "RecoMuon/L2MuonSeedGenerator/src/L2MuonSeedGenerator.h"

// Data Formats 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

#include "CLHEP/Vector/ThreeVector.h"
// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/Surface/interface/BoundCylinder.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

using namespace std;
using namespace edm;
using namespace l1extra;

// constructors
L2MuonSeedGenerator::L2MuonSeedGenerator(const edm::ParameterSet& iConfig) : 
  theSource(iConfig.getParameter<InputTag>("InputObjects")),
  thePropagatorName(iConfig.getParameter<string>("Propagator")),
  theL1MinPt(iConfig.getParameter<double>("L1MinPt")),
  theL1MaxEta(iConfig.getParameter<double>("L1MaxEta")),
  theL1MinQuality(iConfig.getParameter<unsigned int>("L1MinQuality")){
  
  // service parameters
  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);

  // the estimator
  theEstimator = new Chi2MeasurementEstimator(10000.);

  produces<TrajectorySeedCollection>(); 
}

// destructor
L2MuonSeedGenerator::~L2MuonSeedGenerator(){
  if (theService) delete theService;
  if (theEstimator) delete theEstimator;
}

void L2MuonSeedGenerator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  const std::string metname = "Muon|RecoMuon|L2MuonSeedGenerator";
  MuonPatternRecoDumper debug;

  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  // Muon particles
  edm::Handle<L1MuonParticleCollection> muColl;
  iEvent.getByLabel(theSource, muColl);
  LogDebug(metname) << "Number of muons " << muColl->size() << endl;
  
  L1MuonParticleCollection::const_iterator it;
  for(it = muColl->begin(); it != muColl->end(); it++) {
    
    const L1MuGMTCand muonCand = (*it).gmtMuonCand();
    unsigned int quality = 0;

    if ( muonCand.empty() ) {
      LogWarning(metname) << "L2MuonSeedGenerator: WARNING, no L1MuGMTCand! " << endl;
      LogWarning(metname) << "L2MuonSeedGenerator:   this should make sense only within MC tests" << endl;
      // FIXME! Temporary to handle the MC input
      quality = 7;
    }
    else
      quality =  muonCand.quality();
    
    float pt    =  (*it).pt();
    float eta   =  (*it).eta();
    float theta =  2*atan(exp(-eta));
    float phi   =  (*it).phi();      
    int charge  =  (*it).charge();
    bool barrel = !(*it).isForward();

    if ( pt < theL1MinPt || fabs(eta) > theL1MaxEta ) continue;
    
    LogDebug(metname) << "New L2 Muon Seed";
    LogDebug(metname) << "Pt = " << pt << " GeV/c";
    LogDebug(metname) << "eta = " << eta;
    LogDebug(metname) << "theta = " << theta << " rad";
    LogDebug(metname) << "phi = " << phi << " rad";
    LogDebug(metname) << "charge = "<< charge;
    LogDebug(metname) << "In Barrel? = "<< barrel;
    
    if ( quality <= theL1MinQuality ) continue;
    LogDebug(metname) << "quality = "<< quality; 
    
    // Update the services
    theService->update(iSetup);

    const DetLayer *detLayer = 0;
    float radius = 0.;
  
    Hep3Vector vec(0.,1.,0.);
    vec.setTheta(theta);
    vec.setPhi(phi);
	
    // Get the det layer on which the state should be put
    if ( barrel ){
      LogDebug(metname) << "The seed is in the barrel";
      
      // MB2
      DetId id = DTChamberId(0,2,0);
      detLayer = theService->detLayerGeometry()->idToLayer(id);
      LogDebug(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);
      
      const BoundSurface* sur = &(detLayer->surface());
      const BoundCylinder* bc = dynamic_cast<const BoundCylinder*>(sur);

      radius = fabs(bc->radius()/sin(theta));

      LogDebug(metname) << "radius "<<radius;

      if ( pt < 3.5 ) pt = 3.5;
    }
    else { 
      LogDebug(metname) << "The seed is in the endcap";
      
      DetId id;
      // ME2
      if ( theta < Geom::pi()/2. )
	id = CSCDetId(1,2,0,0,0); 
      else
	id = CSCDetId(2,2,0,0,0); 
      
      detLayer = theService->detLayerGeometry()->idToLayer(id);
      LogDebug(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);

      radius = fabs(detLayer->position().z()/cos(theta));      
      
      if( pt < 1.0) pt = 1.0;
    }
        
    vec.setMag(radius);
    
    GlobalPoint pos(vec.x(),vec.y(),vec.z());
      
    GlobalVector mom(pt*cos(phi), pt*sin(phi), pt*cos(theta)/sin(theta));

    GlobalTrajectoryParameters param(pos,mom,charge,&*theService->magneticField());
    AlgebraicSymMatrix mat(5,0);
    
    mat[0][0] = (0.25/pt)*(0.25/pt);  // sigma^2(charge/abs_momentum)
    if ( !barrel ) mat[0][0] = (0.4/pt)*(0.4/pt);
    
    mat[1][1] = 0.05*0.05;        // sigma^2(lambda)
    mat[2][2] = 0.2*0.2;          // sigma^2(phi)
    mat[3][3] = 20.*20.;          // sigma^2(x_transverse))
    mat[4][4] = 20.*20.;          // sigma^2(y_transverse))
    
    CurvilinearTrajectoryError error(mat);

    const FreeTrajectoryState state(param,error);
   
    LogDebug(metname) << "Free trajectory State from the parameters";
    LogDebug(metname) << debug.dumpFTS(state);

    // Propagate the state on the MB2/ME2 surface
    TrajectoryStateOnSurface tsos = theService->propagator(thePropagatorName)->propagate(state, detLayer->surface());
   
    LogDebug(metname) << "State after the propagation on the layer";
    LogDebug(metname) << debug.dumpLayer(detLayer);
    LogDebug(metname) << debug.dumpFTS(state);

    if (tsos.isValid()) {
      // Get the compatible dets on the layer
      std::vector< pair<const GeomDet*,TrajectoryStateOnSurface> > 
	detsWithStates = detLayer->compatibleDets(tsos, 
						  *theService->propagator(thePropagatorName), 
						  *theEstimator);   
      if (detsWithStates.size()){
	TrajectoryStateTransform tsTransform;
	
	TrajectoryStateOnSurface newTSOS = detsWithStates.front().second;
	const GeomDet *newTSOSDet = detsWithStates.front().first;
	
	LogDebug(metname) << "Most compatible det";
	LogDebug(metname) << debug.dumpMuonId(newTSOSDet->geographicalId());

	if (newTSOS.isValid()){

	  LogDebug(metname) << "State on it";
	  LogDebug(metname) << debug.dumpTSOS(newTSOS);
	  
	  // convert the TSOS into a PTSOD
	  PTrajectoryStateOnDet *seedTSOS = tsTransform.persistentState( newTSOS,newTSOSDet->geographicalId().rawId());
	  
	  edm::OwnVector<TrackingRecHit> container;
	  TrajectorySeed* seed = new TrajectorySeed(*seedTSOS,container,alongMomentum);
	  
	  output->push_back(*seed);
	}
      }
    } 
    
  }
  
  iEvent.put(output);
}

