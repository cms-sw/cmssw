//-------------------------------------------------
//
/**  \class L2MuonSeedGenerator
 * 
 *   L2 muon seed generator:
 *   Transform the L1 informations in seeds for the
 *   L2 muon reconstruction
 *
 *
 *   $Date: 2012/03/15 19:21:21 $
 *   $Revision: 1.16 $
 *
 *   \author  A.Everett, R.Bellan, J. Alcaraz
 *
 *    ORCA's author: N. Neumeister 
 */
//
//--------------------------------------------------

// Class Header
#include "RecoMuon/L2MuonSeedGenerator/src/L2MuonSeedGenerator.h"

// Data Formats 
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CLHEP/Vector/ThreeVector.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

using namespace std;
using namespace edm;
using namespace l1extra;

// constructors
L2MuonSeedGenerator::L2MuonSeedGenerator(const edm::ParameterSet& iConfig) : 
  theSource(iConfig.getParameter<InputTag>("InputObjects")),
  theL1GMTReadoutCollection(iConfig.getParameter<InputTag>("GMTReadoutCollection")),
  thePropagatorName(iConfig.getParameter<string>("Propagator")),
  theL1MinPt(iConfig.getParameter<double>("L1MinPt")),
  theL1MaxEta(iConfig.getParameter<double>("L1MaxEta")),
  theL1MinQuality(iConfig.getParameter<unsigned int>("L1MinQuality")),
  useOfflineSeed(iConfig.getUntrackedParameter<bool>("UseOfflineSeed", false)){
  
  // service parameters
  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);

  // the estimator
  theEstimator = new Chi2MeasurementEstimator(10000.);

  if(useOfflineSeed)
    theOfflineSeedLabel = iConfig.getUntrackedParameter<InputTag>("OfflineSeedLabel");

  produces<L2MuonTrajectorySeedCollection>(); 
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

  auto_ptr<L2MuonTrajectorySeedCollection> output(new L2MuonTrajectorySeedCollection());
  
  // Muon particles and GMT readout collection
  edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
  iEvent.getByLabel(theL1GMTReadoutCollection,gmtrc_handle);
  L1MuGMTReadoutRecord const& gmtrr = gmtrc_handle.product()->getRecord(0);

  edm::Handle<L1MuonParticleCollection> muColl;
  iEvent.getByLabel(theSource, muColl);
  LogTrace(metname) << "Number of muons " << muColl->size() << endl;

  edm::Handle<edm::View<TrajectorySeed> > offlineSeedHandle;
  vector<int> offlineSeedMap;
  if(useOfflineSeed) {
    iEvent.getByLabel(theOfflineSeedLabel, offlineSeedHandle);
    LogTrace(metname) << "Number of offline seeds " << offlineSeedHandle->size() << endl;
    offlineSeedMap = vector<int>(offlineSeedHandle->size(), 0);
  }

  L1MuonParticleCollection::const_iterator it;
  L1MuonParticleRef::key_type l1ParticleIndex = 0;

  for(it = muColl->begin(); it != muColl->end(); ++it,++l1ParticleIndex) {
    
    const L1MuGMTExtendedCand muonCand = (*it).gmtMuonCand();
    unsigned int quality = 0;
    bool valid_charge = false;;

    if ( muonCand.empty() ) {
      LogWarning(metname) << "L2MuonSeedGenerator: WARNING, no L1MuGMTCand! " << endl;
      LogWarning(metname) << "L2MuonSeedGenerator:   this should make sense only within MC tests" << endl;
      // FIXME! Temporary to handle the MC input
      quality = 7;
      valid_charge = true;
    }
    else {
      quality =  muonCand.quality();
      valid_charge = muonCand.charge_valid();
    }
    
    float pt    =  (*it).pt();
    float eta   =  (*it).eta();
    float theta =  2*atan(exp(-eta));
    float phi   =  (*it).phi();      
    int charge  =  (*it).charge();
    // Set charge=0 for the time being if the valid charge bit is zero
    if (!valid_charge) charge = 0;
    bool barrel = !(*it).isForward();

    // Get a better eta and charge from regional information
    // Phi has the same resolution in GMT than regionally, is not it?
    if ( !(muonCand.empty()) ) {
      int idx = -1;
      vector<L1MuRegionalCand> rmc;
      if ( !muonCand.isRPC() ) {
            idx = muonCand.getDTCSCIndex();
            if (muonCand.isFwd()) rmc = gmtrr.getCSCCands();
            else rmc = gmtrr.getDTBXCands();
      } else {
            idx = muonCand.getRPCIndex();
            if (muonCand.isFwd()) rmc = gmtrr.getFwdRPCCands();
            else rmc = gmtrr.getBrlRPCCands();
      }
      if (idx>=0) {
            eta = rmc[idx].etaValue();
            //phi = rmc[idx].phiValue();
            // Use this charge if the valid charge bit is zero
            if (!valid_charge) charge = rmc[idx].chargeValue();
      }
    }

    if ( pt < theL1MinPt || fabs(eta) > theL1MaxEta ) continue;
    
    LogTrace(metname) << "New L2 Muon Seed";
    LogTrace(metname) << "Pt = " << pt << " GeV/c";
    LogTrace(metname) << "eta = " << eta;
    LogTrace(metname) << "theta = " << theta << " rad";
    LogTrace(metname) << "phi = " << phi << " rad";
    LogTrace(metname) << "charge = "<< charge;
    LogTrace(metname) << "In Barrel? = "<< barrel;
    
    if ( quality <= theL1MinQuality ) continue;
    LogTrace(metname) << "quality = "<< quality; 
    
    // Update the services
    theService->update(iSetup);

    const DetLayer *detLayer = 0;
    float radius = 0.;
  
    CLHEP::Hep3Vector vec(0.,1.,0.);
    vec.setTheta(theta);
    vec.setPhi(phi);
	
    // Get the det layer on which the state should be put
    if ( barrel ){
      LogTrace(metname) << "The seed is in the barrel";
      
      // MB2
      DetId id = DTChamberId(0,2,0);
      detLayer = theService->detLayerGeometry()->idToLayer(id);
      LogTrace(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);
      
      const BoundSurface* sur = &(detLayer->surface());
      const BoundCylinder* bc = dynamic_cast<const BoundCylinder*>(sur);

      radius = fabs(bc->radius()/sin(theta));

      LogTrace(metname) << "radius "<<radius;

      if ( pt < 3.5 ) pt = 3.5;
    }
    else { 
      LogTrace(metname) << "The seed is in the endcap";
      
      DetId id;
      // ME2
      if ( theta < Geom::pi()/2. )
	id = CSCDetId(1,2,0,0,0); 
      else
	id = CSCDetId(2,2,0,0,0); 
      
      detLayer = theService->detLayerGeometry()->idToLayer(id);
      LogTrace(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);

      radius = fabs(detLayer->position().z()/cos(theta));      
      
      if( pt < 1.0) pt = 1.0;
    }
        
    vec.setMag(radius);
    
    GlobalPoint pos(vec.x(),vec.y(),vec.z());
      
    GlobalVector mom(pt*cos(phi), pt*sin(phi), pt*cos(theta)/sin(theta));

    GlobalTrajectoryParameters param(pos,mom,charge,&*theService->magneticField());
    AlgebraicSymMatrix55 mat;
    
    mat[0][0] = (0.25/pt)*(0.25/pt);  // sigma^2(charge/abs_momentum)
    if ( !barrel ) mat[0][0] = (0.4/pt)*(0.4/pt);

    //Assign q/pt = 0 +- 1/pt if charge has been declared invalid
    if (!valid_charge) mat[0][0] = (1./pt)*(1./pt);
    
    mat[1][1] = 0.05*0.05;        // sigma^2(lambda)
    mat[2][2] = 0.2*0.2;          // sigma^2(phi)
    mat[3][3] = 20.*20.;          // sigma^2(x_transverse))
    mat[4][4] = 20.*20.;          // sigma^2(y_transverse))
    
    CurvilinearTrajectoryError error(mat);

    const FreeTrajectoryState state(param,error);
   
    LogTrace(metname) << "Free trajectory State from the parameters";
    LogTrace(metname) << debug.dumpFTS(state);

    // Propagate the state on the MB2/ME2 surface
    TrajectoryStateOnSurface tsos = theService->propagator(thePropagatorName)->propagate(state, detLayer->surface());
   
    LogTrace(metname) << "State after the propagation on the layer";
    LogTrace(metname) << debug.dumpLayer(detLayer);
    LogTrace(metname) << debug.dumpFTS(state);

    if (tsos.isValid()) {
      // Get the compatible dets on the layer
      std::vector< pair<const GeomDet*,TrajectoryStateOnSurface> > 
	detsWithStates = detLayer->compatibleDets(tsos, 
						  *theService->propagator(thePropagatorName), 
						  *theEstimator);   
      if (detsWithStates.size()){

	TrajectoryStateOnSurface newTSOS = detsWithStates.front().second;
	const GeomDet *newTSOSDet = detsWithStates.front().first;
	
	LogTrace(metname) << "Most compatible det";
	LogTrace(metname) << debug.dumpMuonId(newTSOSDet->geographicalId());

	LogDebug(metname) << "L1 info: Det and State:";
	LogDebug(metname) << debug.dumpMuonId(newTSOSDet->geographicalId());

	if (newTSOS.isValid()){

	  //LogDebug(metname) << "(x, y, z) = (" << newTSOS.globalPosition().x() << ", " 
	  //	              << newTSOS.globalPosition().y() << ", " << newTSOS.globalPosition().z() << ")";
	  LogDebug(metname) << "pos: (r=" << newTSOS.globalPosition().mag() << ", phi=" 
			    << newTSOS.globalPosition().phi() << ", eta=" << newTSOS.globalPosition().eta() << ")";
	  LogDebug(metname) << "mom: (q*pt=" << newTSOS.charge()*newTSOS.globalMomentum().perp() << ", phi=" 
			    << newTSOS.globalMomentum().phi() << ", eta=" << newTSOS.globalMomentum().eta() << ")";

	  //LogDebug(metname) << "State on it";
	  //LogDebug(metname) << debug.dumpTSOS(newTSOS);

	  //PTrajectoryStateOnDet seedTSOS;
	  edm::OwnVector<TrackingRecHit> container;

	  if(useOfflineSeed) {
	    const TrajectorySeed *assoOffseed = 
	      associateOfflineSeedToL1(offlineSeedHandle, offlineSeedMap, newTSOS);

	    if(assoOffseed!=0) {
	      PTrajectoryStateOnDet const & seedTSOS = assoOffseed->startingState();
	      TrajectorySeed::const_iterator 
		tsci  = assoOffseed->recHits().first,
		tscie = assoOffseed->recHits().second;
	      for(; tsci!=tscie; ++tsci) {
		container.push_back(*tsci);
	      }
	      output->push_back(L2MuonTrajectorySeed(seedTSOS,container,alongMomentum,
						     L1MuonParticleRef(muColl,l1ParticleIndex)));
	    }
	    else {
	      // convert the TSOS into a PTSOD
	      PTrajectoryStateOnDet const & seedTSOS = trajectoryStateTransform::persistentState( newTSOS,newTSOSDet->geographicalId().rawId());
	      output->push_back(L2MuonTrajectorySeed(seedTSOS,container,alongMomentum,
						     L1MuonParticleRef(muColl,l1ParticleIndex)));
	    }
	  }
	  else {
	    // convert the TSOS into a PTSOD
	    PTrajectoryStateOnDet const & seedTSOS = trajectoryStateTransform::persistentState( newTSOS,newTSOSDet->geographicalId().rawId());
	    output->push_back(L2MuonTrajectorySeed(seedTSOS,container,alongMomentum,
						   L1MuonParticleRef(muColl,l1ParticleIndex)));
	  }
	}
      }
    }
  }
  
  iEvent.put(output);
}


// FIXME: does not resolve ambiguities yet! 
const TrajectorySeed* L2MuonSeedGenerator::associateOfflineSeedToL1( edm::Handle<edm::View<TrajectorySeed> > & offseeds, 
								     std::vector<int> & offseedMap, 
								     TrajectoryStateOnSurface & newTsos) {

  const std::string metlabel = "Muon|RecoMuon|L2MuonSeedGenerator";
  MuonPatternRecoDumper debugtmp;

  edm::View<TrajectorySeed>::const_iterator offseed, endOffseed = offseeds->end();
  const TrajectorySeed *selOffseed = 0;
  double bestDr = 99999.;
  unsigned int nOffseed(0);
  int lastOffseed(-1);

  for(offseed=offseeds->begin(); offseed!=endOffseed; ++offseed, ++nOffseed) {
    if(offseedMap[nOffseed]!=0) continue;
    GlobalPoint glbPos = theService->trackingGeometry()->idToDet(offseed->startingState().detId())->surface().toGlobal(offseed->startingState().parameters().position());
    GlobalVector glbMom = theService->trackingGeometry()->idToDet(offseed->startingState().detId())->surface().toGlobal(offseed->startingState().parameters().momentum());

    // Preliminary check
    double preDr = deltaR( newTsos.globalPosition().eta(), newTsos.globalPosition().phi(), glbPos.eta(), glbPos.phi() );
    if(preDr > 1.0) continue;

    const FreeTrajectoryState offseedFTS(glbPos, glbMom, offseed->startingState().parameters().charge(), &*theService->magneticField()); 
    TrajectoryStateOnSurface offseedTsos = theService->propagator(thePropagatorName)->propagate(offseedFTS, newTsos.surface());
    LogDebug(metlabel) << "Offline seed info: Det and State" << std::endl;
    LogDebug(metlabel) << debugtmp.dumpMuonId(offseed->startingState().detId()) << std::endl;
    //LogDebug(metlabel) << "(x, y, z) = (" << newTSOS.globalPosition().x() << ", " 
    //	                 << newTSOS.globalPosition().y() << ", " << newTSOS.globalPosition().z() << ")" << std::endl;
    LogDebug(metlabel) << "pos: (r=" << offseedFTS.position().mag() << ", phi=" 
		       << offseedFTS.position().phi() << ", eta=" << offseedFTS.position().eta() << ")" << std::endl;
    LogDebug(metlabel) << "mom: (q*pt=" << offseedFTS.charge()*offseedFTS.momentum().perp() << ", phi=" 
		       << offseedFTS.momentum().phi() << ", eta=" << offseedFTS.momentum().eta() << ")" << std::endl << std::endl;
    //LogDebug(metlabel) << debugtmp.dumpFTS(offseedFTS) << std::endl;

    if(offseedTsos.isValid()) {
      LogDebug(metlabel) << "Offline seed info after propagation to L1 layer:" << std::endl;
      //LogDebug(metlabel) << "(x, y, z) = (" << offseedTsos.globalPosition().x() << ", " 
      //                   << offseedTsos.globalPosition().y() << ", " << offseedTsos.globalPosition().z() << ")" << std::endl;
      LogDebug(metlabel) << "pos: (r=" << offseedTsos.globalPosition().mag() << ", phi=" 
			 << offseedTsos.globalPosition().phi() << ", eta=" << offseedTsos.globalPosition().eta() << ")" << std::endl;
      LogDebug(metlabel) << "mom: (q*pt=" << offseedTsos.charge()*offseedTsos.globalMomentum().perp() << ", phi=" 
			 << offseedTsos.globalMomentum().phi() << ", eta=" << offseedTsos.globalMomentum().eta() << ")" << std::endl << std::endl;
      //LogDebug(metlabel) << debugtmp.dumpTSOS(offseedTsos) << std::endl;
      double newDr = deltaR( newTsos.globalPosition().eta(),     newTsos.globalPosition().phi(), 
			     offseedTsos.globalPosition().eta(), offseedTsos.globalPosition().phi() );
      LogDebug(metlabel) << "   -- DR = " << newDr << std::endl;
      if( newDr<0.3 && newDr<bestDr ) {
	LogDebug(metlabel) << "          --> OK! " << newDr << std::endl << std::endl;
	selOffseed = &*offseed;
	bestDr = newDr;
	offseedMap[nOffseed] = 1;
	if(lastOffseed>-1) offseedMap[lastOffseed] = 0;
	lastOffseed = nOffseed;
      }
      else {
	LogDebug(metlabel) << "          --> Rejected. " << newDr << std::endl << std::endl;
      }
    }
    else {
      LogDebug(metlabel) << "Invalid offline seed TSOS after propagation!" << std::endl << std::endl;
    }
  }

  return selOffseed;
}
