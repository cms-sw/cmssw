#include "RecoParticleFlow/PFBlockProducer/interface/PFSimParticleProducer.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticleFwd.h"

// include files used for reconstructed tracks
// #include "DataFormats/TrackReco/interface/Track.h"
// #include "DataFormats/TrackReco/interface/TrackFwd.h"
// #include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
// #include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
// #include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
// #include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
// #include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"  
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
// #include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"

#include <set>

using namespace std;
using namespace edm;

PFSimParticleProducer::PFSimParticleProducer(const edm::ParameterSet& iConfig) 
{


  processParticles_ = 
    iConfig.getUntrackedParameter<bool>("process_Particles",true);
    

  simModuleLabel_ 
    = iConfig.getUntrackedParameter<string>
    ("SimModuleLabel","g4SimHits");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);


  // register products
  produces<reco::PFSimParticleCollection>();
  
  vertexGenerator_ = iConfig.getParameter<ParameterSet>
    ( "VertexGenerator" );   
  particleFilter_ = iConfig.getParameter<ParameterSet>
    ( "ParticleFilter" );   
  
  mySimEvent =  new FSimEvent( particleFilter_ );
}



PFSimParticleProducer::~PFSimParticleProducer() 
{ 
  delete mySimEvent; 
}


void 
PFSimParticleProducer::beginJob(const edm::EventSetup & es)
{
  
  // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  //  edm::ESHandle < DefaultConfig::ParticleDataTable > pdt;
  es.getData(pdt);
  if ( !ParticleTable::instance() ) ParticleTable::instance(&(*pdt));
  mySimEvent->initializePdt(&(*pdt));
}


void PFSimParticleProducer::produce(Event& iEvent, 
			 const EventSetup& iSetup) 
{
  
  LogDebug("PFSimParticleProducer")<<"START event: "<<iEvent.id().event()
			<<" in run "<<iEvent.id().run()<<endl;
  
  
   
  // deal with true particles 
  if( processParticles_) {

    try {
      auto_ptr< reco::PFSimParticleCollection > 
	pOutputPFSimParticleCollection(new reco::PFSimParticleCollection ); 

      Handle<vector<SimTrack> > simTracks;
      iEvent.getByLabel(simModuleLabel_,simTracks);
      Handle<vector<SimVertex> > simVertices;
      iEvent.getByLabel(simModuleLabel_,simVertices);

      //     for(unsigned it = 0; it<simTracks->size(); it++ ) {
      //       cout<<"\t track "<< (*simTracks)[it]<<" "
      // 	  <<(*simTracks)[it].momentum().vect().perp()<<" "
      // 	  <<(*simTracks)[it].momentum().e()<<endl;
      //     }

      mySimEvent->fill( *simTracks, *simVertices );
      
      if(verbose_) 
	mySimEvent->print();

      // const std::vector<FSimTrack>& fsimTracks = *(mySimEvent->tracks() );
      for(unsigned i=0; i<mySimEvent->nTracks(); i++) {
    
	const FSimTrack& fst = mySimEvent->track(i);

	int motherId = -1;
	if( ! fst.noMother() ) // a mother exist
	  motherId = fst.mother().id();

	reco::PFSimParticle particle(  fst.charge(), 
				       fst.type(), 
				       fst.id(), 
				       motherId,
				       fst.daughters() );


	const FSimVertex& originVtx = fst.vertex();

	math::XYZPoint          posOrig( originVtx.position().x(), 
					 originVtx.position().y(), 
					 originVtx.position().z() );

	math::XYZTLorentzVector momOrig( fst.momentum().px(), 
					 fst.momentum().py(), 
					 fst.momentum().pz(), 
					 fst.momentum().e() );
	reco::PFTrajectoryPoint 
	  pointOrig(-1, 
		    reco::PFTrajectoryPoint::ClosestApproach,
		    posOrig, momOrig);

	// point 0 is origin vertex
	particle.addPoint(pointOrig);
    

	if( ! fst.noEndVertex() ) {
	  const FSimVertex& endVtx = fst.endVertex();
	
	  math::XYZPoint          posEnd( endVtx.position().x(), 
					  endVtx.position().y(), 
					  endVtx.position().z() );
	  //       cout<<"end vertex : "
	  // 	  <<endVtx.position().x()<<" "
	  // 	  <<endVtx.position().y()<<endl;
	
	  math::XYZTLorentzVector momEnd;
	
	  reco::PFTrajectoryPoint 
	    pointEnd( -1, 
		      reco::PFTrajectoryPoint::BeamPipeOrEndVertex,
		      posEnd, momEnd);
	
	  particle.addPoint(pointEnd);
	}
	else { // add a dummy point
	  reco::PFTrajectoryPoint dummy;
	  particle.addPoint(dummy); 
	}


	if( fst.onLayer1() ) { // PS layer1
	  const RawParticle& rp = fst.layer1Entrance();
      
	  math::XYZPoint posLayer1( rp.x(), rp.y(), rp.z() );
	  math::XYZTLorentzVector momLayer1( rp.px(), rp.py(), rp.pz(), rp.e() );
	  reco::PFTrajectoryPoint layer1Pt(-1, reco::PFTrajectoryPoint::PS1, 
					   posLayer1, momLayer1);
	
	  particle.addPoint( layer1Pt ); 

	  // extrapolate to cluster depth
	}
	else { // add a dummy point
	  reco::PFTrajectoryPoint dummy;
	  particle.addPoint(dummy); 
	}

	if( fst.onLayer2() ) { // PS layer2
	  const RawParticle& rp = fst.layer2Entrance();
      
	  math::XYZPoint posLayer2( rp.x(), rp.y(), rp.z() );
	  math::XYZTLorentzVector momLayer2( rp.px(), rp.py(), rp.pz(), rp.e() );
	  reco::PFTrajectoryPoint layer2Pt(-1, reco::PFTrajectoryPoint::PS2, 
					   posLayer2, momLayer2);
	
	  particle.addPoint( layer2Pt ); 

	  // extrapolate to cluster depth
	}
	else { // add a dummy point
	  reco::PFTrajectoryPoint dummy;
	  particle.addPoint(dummy); 
	}

	if( fst.onEcal() ) {
	  const RawParticle& rp = fst.ecalEntrance();
	
	  math::XYZPoint posECAL( rp.x(), rp.y(), rp.z() );
	  math::XYZTLorentzVector momECAL( rp.px(), rp.py(), rp.pz(), rp.e() );
	  reco::PFTrajectoryPoint ecalPt(-1, 
					 reco::PFTrajectoryPoint::ECALEntrance, 
					 posECAL, momECAL);
	
	  particle.addPoint( ecalPt ); 

	  // extrapolate to cluster depth
	}
	else { // add a dummy point
	  reco::PFTrajectoryPoint dummy;
	  particle.addPoint(dummy); 
	}
	
	// add a dummy point for ECAL Shower max
	reco::PFTrajectoryPoint dummy;
	particle.addPoint(dummy); 

	if( fst.onHcal() ) {

	  const RawParticle& rpin = fst.hcalEntrance();
	
	  math::XYZPoint posHCALin( rpin.x(), rpin.y(), rpin.z() );
	  math::XYZTLorentzVector momHCALin( rpin.px(), rpin.py(), rpin.pz(), 
					     rpin.e() );
	  reco::PFTrajectoryPoint hcalPtin(-1, 
					   reco::PFTrajectoryPoint::HCALEntrance,
					   posHCALin, momHCALin);
	
	  particle.addPoint( hcalPtin ); 


	  // 	const RawParticle& rpout = fst.hcalExit();
	
	  // 	math::XYZPoint posHCALout( rpout.x(), rpout.y(), rpout.z() );
	  // 	math::XYZTLorentzVector momHCALout( rpout.px(), rpout.py(), rpout.pz(),
	  //  					    rpout.e() );
	  // 	reco::PFTrajectoryPoint 
	  // 	  hcalPtout(0, reco::PFTrajectoryPoint::HCALEntrance, 
	  // 		    posHCAL, momHCAL);
	
	  // 	particle.addPoint( hcalPtout ); 	
	}
	else { // add a dummy point
	  reco::PFTrajectoryPoint dummy;
	  particle.addPoint(dummy); 
	}
          
	pOutputPFSimParticleCollection->push_back( particle );
      }
      
      iEvent.put(pOutputPFSimParticleCollection);
    }
    catch(cms::Exception& err) { 
      LogError("PFSimParticleProducer")<<err.what()<<endl;
    }
  }

  LogDebug("PFSimParticleProducer")<<"STOP event: "<<iEvent.id().event()
			<<" in run "<<iEvent.id().run()<<endl;
}

