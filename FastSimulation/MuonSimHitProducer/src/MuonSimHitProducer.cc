//
// Package:    MuonSimHitProducer
// Class:      MuonSimHitProducer
// 
/**\class MuonSimHitProducer FastSimulation/MuonSimHitProducer/src/MuonSimHitProducer.cc

 Description:
    Fast simulation producer of Muon Sim Hits (to be used for realistic Muon reconstruction)

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Martijn Mulders
//         Created:  Wed Jul 30 11:37:24 CET 2007
// $Id: MuonSimHitProducer.cc,v 1.0 2007/07/11 13:53:50 mulders Exp $
//
//

// CMSSW headers 
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/PluginManager.h"

// Fast Simulation headers
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/MuonSimHitProducer/interface/MuonSimHitProducer.h"

// SimTrack
#include "SimDataFormats/Track/interface/SimTrack.h"

// STL headers 
#include <vector>
#include <iostream>

// CLHEP headers
#include "DataFormats/Math/interface/LorentzVector.h"

// Data Formats
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"

////////////////////////////////////////////////////////////////////////////

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include  "TrackPropagation/NavPropagator/interface/NavPropagator.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// constants, enums and typedefs
typedef std::vector<L1MuGMTCand> L1MuonCollection;


//
// static data member definitions
//

//
// constructors and destructor
//
MuonSimHitProducer::MuonSimHitProducer(const edm::ParameterSet& iConfig)
{

  readParameters(iConfig.getParameter<edm::ParameterSet>("MUONS"),
		 iConfig.getParameter<edm::ParameterSet>("TRACKS"));

  //register your products ... need to declare at least one possible product...
    if (doL1_) produces<std::vector<L1MuGMTCand> >("HitL1Muons");
    if (doL3_) produces<reco::MuonCollection>("HitL3Muons");
    if (doGL_) produces<reco::MuonCollection>("HitGlobalMuons");

  // Initialize the random number generator service
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration") <<
      "ParamMuonProducer requires the RandomGeneratorService \n"
      "which is not present in the configuration file. \n"
      "You must add the service in the configuration file\n"
      "or remove the module that requires it.";
  }
  random = new RandomEngine(&(*rng));

}


MuonSimHitProducer::~MuonSimHitProducer()
{
 
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  
  if ( random ) delete random;
}


//
// member functions
//

// ------------ method called to produce the data  ------------

void MuonSimHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  Handle<std::vector<SimTrack> > simMuons;
  iEvent.getByLabel(theSimModuleLabel_,theSimModuleProcess_,simMuons);
  unsigned nmuons = simMuons->size();


  for( unsigned fsimi=0; fsimi < nmuons; ++fsimi) {
    const SimTrack& mySimTrack = (*simMuons)[fsimi];

    bool hasL1 = false , hasL3 = false , hasTK = false , hasGL = false;
    //Replace with this as soon transition to ROOTMath is complete
    //    math::XYZTLorentzVector& mySimP4 =  mySimTrack.momentum();

    math::XYZTLorentzVector mySimP4 =  math::XYZTLorentzVector(mySimTrack.momentum().x(),
							       mySimTrack.momentum().y(),
							       mySimTrack.momentum().z(),
							       mySimTrack.momentum().t());

    std::cout << " AND THIS IS MUON NO. " << fsimi << std::endl;
    if (debug_) {
      std::cout << " ===> ParamMuonProducer::reconstruct() found SIMTRACK - pid = "
		<< mySimTrack.type() ;
      std::cout << " : pT = " << mySimP4.Pt()
		<< ", eta = " << mySimP4.Eta()
		<< ", phi = " << mySimP4.Phi() << std::endl;

    }

// *** Reconstruct parameterized muons starting from undecayed simulated muons
 
    GlobalPoint startingPosition = GlobalPoint(mySimTrack.trackerSurfacePosition().x(),
					       mySimTrack.trackerSurfacePosition().y(),
					       mySimTrack.trackerSurfacePosition().z());
    GlobalVector startingMomentum = GlobalVector(mySimTrack.trackerSurfaceMomentum().x(),
						 mySimTrack.trackerSurfaceMomentum().y(),
						 mySimTrack.trackerSurfaceMomentum().z());
    std::cout << " the Muon START position " << startingPosition << std::endl;
    std::cout << " the Muon START momentum " << startingMomentum << std::endl;

    // Get access to the magnetic field
    edm::ESHandle<MagneticField> magfield;
    try
      {
	iSetup.get<IdealMagneticFieldRecord>().get(magfield);
      }
    catch (...)
      {
	std::cout << "FAILED: to get the IdealMagneticFieldRecord!" << std::endl;
      }
    

    // Some magic to define a TrajectoryStateOnSurface
    PlaneBuilder pb;
    GlobalVector zAxis = startingMomentum.unit();
    GlobalVector yAxis( zAxis.y(), -zAxis.x(), 0); 
    GlobalVector xAxis = yAxis.cross( zAxis);
    Surface::RotationType rot = Surface::RotationType( xAxis, yAxis, zAxis);
    PlaneBuilder::ReturnType startingPlane = pb.plane( startingPosition, rot);
    GlobalTrajectoryParameters gtp (startingPosition, startingMomentum, -1, magfield.product() );
    TrajectoryStateOnSurface startingState( gtp, *startingPlane);

    // Define destination plane, 1 meter away in the direction of the muon momentum:
    float propDistance = 100;
    //GlobalPoint targetPos ((propDistance*startingMomentum.unit() + mySimTrack.trackerSurfacePosition());
    //    targetPos = targetPos + startingPosition;
    GlobalVector PropVector = propDistance*startingMomentum.unit();
    GlobalPoint targetPos = startingPosition + PropVector;
    PlaneBuilder::ReturnType muonPlane = pb.plane( targetPos , rot);

    // Get a propagator
    NavPropagator prop(magfield.product());
    
    // Do the actual propagation:
    std::cout << "Propagating toward muon system ................ " << std::endl;
    TrajectoryStateOnSurface FinalState = prop.propagate( startingState, *muonPlane);
    if (FinalState.isValid()) {
      std::cout << "Yes! this muon reached final destination at position " << FinalState.globalPosition() << std::endl;
      std::cout << "which corresponds to an eta of " << std::endl;
      nMuonTot++;
    } else {
      std::cout << "Oops, this muon got lost" << std::endl;
    }
    
 
      
  } // end of loop over generated muons


}



// ------------ method called once each job just before starting event loop  ------------
void MuonSimHitProducer::beginJob(const edm::EventSetup& es)
{

  // Initialize
  nMuonTot = 0;

}


// ------------ method called once each job just after ending the event loop  ------------
void MuonSimHitProducer::endJob() {

  std::cout << " ===> MuonSimHitProducer , final report." << std::endl;
  std::cout << " ===> Number of succesfully propagated muons in the whole run : "
  <<   nMuonTot << " -> " << std::endl;
}


void MuonSimHitProducer::readParameters(const edm::ParameterSet& fastMuons, 
					 const edm::ParameterSet& fastTracks) {
  // Muons
  debug_ = fastMuons.getUntrackedParameter<bool>("Debug");
  doL1_ = fastMuons.getUntrackedParameter<bool>("ProduceL1Muons");
  doL3_ = fastMuons.getUntrackedParameter<bool>("ProduceL3Muons");
  doGL_ = fastMuons.getUntrackedParameter<bool>("ProduceGlobalMuons");
  theSimModuleLabel_ = fastMuons.getParameter<std::string>("simModuleLabel");
  theSimModuleProcess_ = fastMuons.getParameter<std::string>("simModuleProcess");
  theTrkModuleLabel_ = fastMuons.getParameter<std::string>("trackModuleLabel");
  minEta_ = fastMuons.getParameter<double>("MinEta");
  maxEta_ = fastMuons.getParameter<double>("MaxEta");
  if (minEta_ > maxEta_) {
    double tempEta_ = maxEta_ ;
    maxEta_ = minEta_ ;
    minEta_ = tempEta_ ;
  }

  // Tracks
  fullPattern_  = fastTracks.getUntrackedParameter<bool>("FullPatternRecognition");

  std::cout << " MUON SIM HITS: FastSimulation parameters " << std::endl;
  std::cout << " ============================================== " << std::endl;
  std::cout << " Sim Hits produced for muons in the pseudorapidity range : "
            << minEta_ << " -> " << maxEta_ << std::endl;
  if ( fullPattern_ ) 
    std::cout << " The FULL pattern recognition option is turned ON" << std::endl;
  else
    std::cout << " The FAST tracking option is turned ON" << std::endl;
}


//define this as a plug-in
DEFINE_FWK_MODULE(MuonSimHitProducer);
