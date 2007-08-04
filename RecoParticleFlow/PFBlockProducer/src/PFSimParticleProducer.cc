#include "RecoParticleFlow/PFBlockProducer/interface/PFSimParticleProducer.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticleFwd.h"

// include files used for reconstructed tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

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

PFSimParticleProducer::PFSimParticleProducer(const edm::ParameterSet& iConfig) :
  trackAlgo_(iConfig) {

  
  processRecTracks_ = 
    iConfig.getUntrackedParameter<bool>("process_RecTracks",true);

  processParticles_ = 
    iConfig.getUntrackedParameter<bool>("process_Particles",true);
    
  

  // use configuration file to setup input/output collection names
  recTrackModuleLabel_ 
    = iConfig.getUntrackedParameter<string>
    ("RecTrackModuleLabel","ckfTrackCandidates");


  simModuleLabel_ 
    = iConfig.getUntrackedParameter<string>
    ("SimModuleLabel","g4SimHits");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);


  // register products
  produces<reco::PFSimParticleCollection>();

  if(processRecTracks_ ) {
    
    produces<reco::PFRecTrackCollection>();
    
    
    // initialize track reconstruction ------------------------------
    fitterName_ = iConfig.getParameter<string>("Fitter");   
    propagatorName_ = iConfig.getParameter<string>("Propagator");
    builderName_ = iConfig.getParameter<string>("TTRHBuilder");   
    
    
    // initialize geometry parameters
    //  PFGeometry pfGeometry;
  }
  
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
  
  
  
  // output collection for rectracks. will be used for particle flow
  

  // deal with RecTracks
  if(processRecTracks_) {
    auto_ptr< reco::PFRecTrackCollection > 
      pOutputPFRecTrackCollection(new reco::PFRecTrackCollection);
    

    processRecTracks(pOutputPFRecTrackCollection,
		     iEvent, iSetup);


    iEvent.put(pOutputPFRecTrackCollection);
  }

  
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


void 
PFSimParticleProducer::processRecTracks(auto_ptr< reco::PFRecTrackCollection >& 
			     trackCollection, 
			     Event& iEvent, 
			     const EventSetup& iSetup) {
  
      // Declare and get stuff from event setup 
    ESHandle<TrackerGeometry> theG;
    ESHandle<MagneticField> theMF;
    ESHandle<TrajectoryFitter> theFitter;
    ESHandle<Propagator> thePropagator;
    ESHandle<TransientTrackingRecHitBuilder> theBuilder;
    try {
      LogDebug("PFSimParticleProducer")<<"get tracker geometry"<<endl;
      iSetup.get<TrackerDigiGeometryRecord>().get(theG);

      LogDebug("PFSimParticleProducer")<<"get magnetic field"<<endl;
      iSetup.get<IdealMagneticFieldRecord>().get(theMF);
      
      LogDebug("PFSimParticleProducer")<<"get the trajectory fitter from the ES"<<endl;
      iSetup.get<TrackingComponentsRecord>().get(fitterName_, theFitter);
      
      LogDebug("PFSimParticleProducer")<<"get the trajectory propagator from the ES"<<endl;
      iSetup.get<TrackingComponentsRecord>().get(propagatorName_, thePropagator);
      
      LogDebug("PFSimParticleProducer")<<"get the TransientTrackingRecHitBuilder"<<endl;
      iSetup.get<TransientRecHitRecord>().get(builderName_, theBuilder);
    }
    catch( exception& err ) {
      LogError("PFSimParticleProducer")
	<<"PFSimParticleProducer::processRecTracks : exception: "
	<<err.what()<<"\n"
	<<"Processing of tracks skipped for this event.\n"
	<<"Note that it might not be a problem for you,\n"
	<<"if you're only interested in clustering.\n";

      return; 
    }



    //
    // Prepare propagation tools and layers
    //
    const MagneticField * magField = theMF.product();

    AnalyticalPropagator fwdPropagator(magField, alongMomentum);

    AnalyticalPropagator bkwdPropagator(magField, oppositeToMomentum);


    // Get track candidates and create smoothed tracks
    // Temporary solution. Let's hope that in the future, Trajectory objects 
    // will become persistent.
    AlgoProductCollection algoResults;
    try{
      LogDebug("PFSimParticleProducer")<<"get the TrackCandidateCollection"
			    <<" from the event, source is " 
			    <<recTrackModuleLabel_ <<endl;
      Handle<TrackCandidateCollection> theTCCollection;
      iEvent.getByLabel(recTrackModuleLabel_, theTCCollection);

      //run the algorithm  
      LogDebug("PFSimParticleProducer")<<"run the tracking algorithm"<<endl;
      trackAlgo_.runWithCandidate(theG.product(), theMF.product(), 
				  *theTCCollection,
				  theFitter.product(), thePropagator.product(),
				  theBuilder.product(), algoResults);

      
      assert( algoResults.size() == theTCCollection->size() );

    } catch (cms::Exception& e) { 
      LogError("PFSimParticleProducer")<<"cms::Exception caught : " 
			    << "cannot get collection " 
			    << recTrackModuleLabel_<<endl<<e<<endl;
    }

    // Loop over smoothed tracks and fill PFRecTrack collection 
    for(AlgoProductCollection::iterator itTrack = algoResults.begin();
	itTrack != algoResults.end(); itTrack++) {
      Trajectory*  theTraj  = (*itTrack).first;
      vector<TrajectoryMeasurement> measurements = theTraj->measurements();

      //      reco::Track* theTrack = (*itTrack).second.first;
      reco::Track* theTrack = (*itTrack).second;

      reco::PFRecTrack track(theTrack->charge(), 
			     reco::PFRecTrack::KF);
      int side = 100;

      // Closest approach of the beamline
      math::XYZPoint posClosest(theTrack->vx(), theTrack->vy(), theTrack->vz());
      math::XYZTLorentzVector momClosest(theTrack->px(), theTrack->py(), 
					 theTrack->pz(), theTrack->p());
      reco::PFTrajectoryPoint 
	closestPt(-1, 
		  reco::PFTrajectoryPoint::ClosestApproach,
		  posClosest, momClosest);
      track.addPoint(closestPt);
      LogDebug("PFSimParticleProducer")<<"closest approach point "<<closestPt<<endl;
    
      if (posClosest.Rho() < PFGeometry::innerRadius(PFGeometry::BeamPipe)) {
	// Intersection with beam pipe

	TrajectoryStateOnSurface innerTSOS;
	if (theTraj->direction() == alongMomentum)
	  innerTSOS = measurements[0].updatedState();
	else
	  innerTSOS = measurements[measurements.size() - 1].updatedState();

	TrajectoryStateOnSurface beamPipeTSOS = 
	  getStateOnSurface(PFGeometry::BeamPipeWall, innerTSOS, 
			    bkwdPropagator, side);
	  //	  bkwdPropagator.propagate(innerTSOS, *beamPipe_);
	
	
	// invalid TSOS, skip track
	if(!beamPipeTSOS.isValid() ) continue;  


	GlobalPoint vBeamPipe  = beamPipeTSOS.globalParameters().position();
	GlobalVector pBeamPipe = beamPipeTSOS.globalParameters().momentum();
	math::XYZPoint posBeamPipe(vBeamPipe.x(), 
				   vBeamPipe.y(), 
				   vBeamPipe.z());
	math::XYZTLorentzVector momBeamPipe(pBeamPipe.x(), 
					    pBeamPipe.y(), 
					    pBeamPipe.z(), 
					    pBeamPipe.mag());
	reco::PFTrajectoryPoint beamPipePt(-1, 
					   reco::PFTrajectoryPoint::BeamPipeOrEndVertex, 
					   posBeamPipe, momBeamPipe);
	
	track.addPoint(beamPipePt);
	LogDebug("PFSimParticleProducer")<<"beam pipe point "<<beamPipePt<<endl;
      }

      // Loop over trajectory measurements

      // Order measurements along momentum
      // COLIN: ?? along R should be better ?
      int iTrajFirst = 0;
      int iTrajLast  = measurements.size();
      int increment = +1;
      if (theTraj->direction() == oppositeToMomentum) {
	iTrajFirst = measurements.size() - 1;
	iTrajLast = -1;
	increment = -1;
      }
      for (int iTraj = iTrajFirst; iTraj != iTrajLast; iTraj += increment) {
	TrajectoryStateOnSurface tsos = measurements[iTraj].updatedState();
	GlobalPoint v  = tsos.globalParameters().position();
	GlobalVector p = tsos.globalParameters().momentum();
	math::XYZPoint  pos(v.x(), v.y(), v.z());       
	math::XYZTLorentzVector mom(p.x(), p.y(), p.z(), p.mag());
	unsigned int detId = 
	  measurements[iTraj].recHit()->det()->geographicalId().rawId();
	reco::PFTrajectoryPoint trajPt(detId, -1, 
				       pos, mom);
	track.addPoint(trajPt);
	LogDebug("PFSimParticleProducer")<<"add measuremnt "<<iTraj<<" "<<trajPt<<endl;
      }

      // Propagate track to ECAL entrance
      TrajectoryStateOnSurface outerTSOS;
      if (theTraj->direction() == alongMomentum)
	outerTSOS = measurements[measurements.size() - 1].updatedState();
      else
	outerTSOS = measurements[0].updatedState();
      int ecalSide = 100;
      TrajectoryStateOnSurface ecalTSOS = 
	getStateOnSurface(PFGeometry::ECALInnerWall, outerTSOS, 
			  fwdPropagator, ecalSide);

      // invalid TSOS, skip track
      if(!ecalTSOS.isValid() ) continue;  

      //fwdPropagator.propagate(outerTSOS, *ecalInnerWall_);
      GlobalPoint vECAL  = ecalTSOS.globalParameters().position();
      GlobalVector pECAL = ecalTSOS.globalParameters().momentum();
      math::XYZPoint posECAL(vECAL.x(), vECAL.y(), vECAL.z());       
      math::XYZTLorentzVector momECAL(pECAL.x(), pECAL.y(), pECAL.z(), 
				      pECAL.mag());
      reco::PFTrajectoryPoint ecalPt(-1, reco::PFTrajectoryPoint::ECALEntrance, 
				     posECAL, momECAL);
      bool isBelowPS = false;
      if (posECAL.Rho() < PFGeometry::innerRadius(PFGeometry::ECALBarrel)) {
	// Propagate track to preshower layer1
	TrajectoryStateOnSurface ps1TSOS = 
	  getStateOnSurface(PFGeometry::PS1Wall, outerTSOS, 
			    fwdPropagator, side);

	// invalid TSOS, skip track
	if(! ps1TSOS.isValid() ) continue;  
	
	//  fwdPropagator.propagate(outerTSOS, *ps1Wall_);
	GlobalPoint vPS1  = ps1TSOS.globalParameters().position();
	GlobalVector pPS1 = ps1TSOS.globalParameters().momentum();
	math::XYZPoint posPS1(vPS1.x(), vPS1.y(), vPS1.z());
	if (posPS1.Rho() >= PFGeometry::innerRadius(PFGeometry::PS1) &&
	    posPS1.Rho() <= PFGeometry::outerRadius(PFGeometry::PS1)) {
	  isBelowPS = true;
	  math::XYZTLorentzVector momPS1(pPS1.x(), pPS1.y(), pPS1.z(), 
					 pPS1.mag());
	  reco::PFTrajectoryPoint ps1Pt(-1, reco::PFTrajectoryPoint::PS1, 
					posPS1, momPS1);
	  track.addPoint(ps1Pt);
	  LogDebug("PFSimParticleProducer")<<"ps1 point "<<ps1Pt<<endl;
	} else {
	  reco::PFTrajectoryPoint dummyPS1;
	  track.addPoint(dummyPS1);
	}

	// Propagate track to preshower layer2
	TrajectoryStateOnSurface ps2TSOS = 
	  getStateOnSurface(PFGeometry::PS2Wall, outerTSOS, 
			    fwdPropagator, side);

	// invalid TSOS, skip track
	if(! ps2TSOS.isValid() ) continue;  

	//  fwdPropagator.propagate(outerTSOS, *ps2Wall_);
	GlobalPoint vPS2  = ps2TSOS.globalParameters().position();
	GlobalVector pPS2 = ps2TSOS.globalParameters().momentum();
	math::XYZPoint posPS2(vPS2.x(), vPS2.y(), vPS2.z());
	if (posPS2.Rho() >= PFGeometry::innerRadius(PFGeometry::PS2) &&
	    posPS2.Rho() <= PFGeometry::outerRadius(PFGeometry::PS2)) {
	  isBelowPS = true;
	  math::XYZTLorentzVector momPS2(pPS2.x(), pPS2.y(), pPS2.z(), 
					 pPS2.mag());
	  reco::PFTrajectoryPoint ps2Pt(-1, reco::PFTrajectoryPoint::PS2, 
					posPS2, momPS2);
	  track.addPoint(ps2Pt);
	  LogDebug("PFSimParticleProducer")<<"ps2 point "<<ps2Pt<<endl;
	} else {
	  reco::PFTrajectoryPoint dummyPS2;
	  track.addPoint(dummyPS2);
	}
      } else {
	// add dummy point for PS1 and PS2
	reco::PFTrajectoryPoint dummyPS1;
	reco::PFTrajectoryPoint dummyPS2;
	track.addPoint(dummyPS1);
	track.addPoint(dummyPS2);
      }
      track.addPoint(ecalPt);
      LogDebug("PFSimParticleProducer")<<"ecal point "<<ecalPt<<endl;

      // Propage track to ECAL shower max TODO
      // Be careful : the following formula are only valid for electrons !
      double ecalShowerDepth 
	= reco::PFCluster::getDepthCorrection(momECAL.E(), 
					      isBelowPS, 
					      false);

      math::XYZPoint showerDirection(momECAL.Px(), momECAL.Py(), momECAL.Pz());
      showerDirection *= ecalShowerDepth/showerDirection.R();
      double rCyl = PFGeometry::innerRadius(PFGeometry::ECALBarrel) + 
	showerDirection.Rho();
      double zCyl = PFGeometry::innerZ(PFGeometry::ECALEndcap) + 
	fabs(showerDirection.Z());
      ReferenceCountingPointer<Surface> showerMaxWall;
      const float epsilon = 0.001; // should not matter at all
      switch (side) {
      case 0: 
	showerMaxWall 
	  = ReferenceCountingPointer<Surface>( new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(rCyl, rCyl, -zCyl, zCyl))); 
	break;
      case +1: 
	showerMaxWall 
	  = ReferenceCountingPointer<Surface>( new BoundPlane(Surface::PositionType(0,0,zCyl), TkRotation<float>(), SimpleDiskBounds(0., rCyl, -epsilon, epsilon))); 
	break;
      case -1: 
	showerMaxWall 
	  = ReferenceCountingPointer<Surface>(new BoundPlane(Surface::PositionType(0,0,-zCyl), TkRotation<float>(), SimpleDiskBounds(0., rCyl, -epsilon, epsilon))); 
	break;
      }
      
      
      TrajectoryStateOnSurface showerMaxTSOS = 
	fwdPropagator.propagate(ecalTSOS, *showerMaxWall);

      if( !showerMaxTSOS.isValid() ) continue;
      GlobalPoint vShowerMax  = showerMaxTSOS.globalParameters().position();
      GlobalVector pShowerMax = showerMaxTSOS.globalParameters().momentum();
      math::XYZPoint posShowerMax(vShowerMax.x(), vShowerMax.y(), 
				  vShowerMax.z());
      math::XYZTLorentzVector momShowerMax(pShowerMax.x(), pShowerMax.y(), 
					   pShowerMax.z(), pShowerMax.mag());
      reco::PFTrajectoryPoint eSMaxPt(-1, 
				      reco::PFTrajectoryPoint::ECALShowerMax, 
				      posShowerMax, momShowerMax);
      track.addPoint(eSMaxPt);
      LogDebug("PFSimParticleProducer")<<"ecal shower maximum point "<<eSMaxPt 
				       <<endl;    



    
      // Propagate track to HCAL entrance

      try {
	TrajectoryStateOnSurface hcalTSOS = 
	  getStateOnSurface(PFGeometry::HCALInnerWall, ecalTSOS, 
			    fwdPropagator, side);

	// invalid TSOS, skip track
	if(! hcalTSOS.isValid() ) continue;  

	//  fwdPropagator.propagate(ecalTSOS, *hcalInnerWall_);
	GlobalPoint vHCAL  = hcalTSOS.globalParameters().position();
	GlobalVector pHCAL = hcalTSOS.globalParameters().momentum();
	math::XYZPoint posHCAL(vHCAL.x(), vHCAL.y(), vHCAL.z());       
	math::XYZTLorentzVector momHCAL(pHCAL.x(), pHCAL.y(), pHCAL.z(), 
					pHCAL.mag());
	reco::PFTrajectoryPoint hcalPt(-1, 
				       reco::PFTrajectoryPoint::HCALEntrance, 
				       posHCAL, momHCAL);
	track.addPoint(hcalPt);
	LogDebug("PFSimParticleProducer")<<"hcal point "<<hcalPt<<endl;    

	// Propagate track to HCAL exit
	// 	TrajectoryStateOnSurface hcalExitTSOS = 
	// 	  fwdPropagator.propagate(hcalTSOS, *hcalOuterWall_);
	// 	GlobalPoint vHCALExit  = hcalExitTSOS.globalParameters().position();
	// 	GlobalVector pHCALExit = hcalExitTSOS.globalParameters().momentum();
	// 	math::XYZPoint posHCALExit(vHCALExit.x(), vHCALExit.y(), vHCALExit.z());
	// 	math::XYZTLorentzVector momHCALExit(pHCALExit.x(), pHCALExit.y(), 
	// 					    pHCALExit.z(), pHCALExit.mag());
	// 	reco::PFTrajectoryPoint hcalExitPt(0, reco::PFTrajectoryPoint::HCALExit, 
	// 					   posHCALExit, momHCALExit);
	// 	track.addPoint(hcalExitPt);
	// 	LogDebug("PFSimParticleProducer")<<"hcal exit point "<<hcalExitPt<<endl;    
      }
      catch( exception& err) {
	LogError("PFSimParticleProducer")<<"Exception : "<<err.what()<<endl;
	throw err; 
      }


      trackCollection->push_back(track);
      LogDebug("PFSimParticleProducer")<<"PFRecTrack added to event"<<track<<endl;
    }   
}



TrajectoryStateOnSurface 
PFSimParticleProducer::getStateOnSurface( PFGeometry::Surface_t iSurf, 
			       const TrajectoryStateOnSurface& tsos, 
			       const Propagator& propagator, int& side) {

  GlobalVector p = tsos.globalParameters().momentum();
  TrajectoryStateOnSurface finalTSOS;
  side = -100;
  if (fabs(p.perp()/p.z()) > PFGeometry::tanTh(iSurf)) {
    finalTSOS = propagator.propagate(tsos, PFGeometry::barrelBound(iSurf));
    side = 0;
    if (!finalTSOS.isValid()) {
      if (p.z() > 0.) {
	finalTSOS = propagator.propagate(tsos, PFGeometry::positiveEndcapDisk(iSurf));
	side = 1;
      } else {
	finalTSOS = propagator.propagate(tsos, PFGeometry::negativeEndcapDisk(iSurf));
	side = -1;
      }
    }
  } else if (p.z() > 0.) {
    finalTSOS = propagator.propagate(tsos, PFGeometry::positiveEndcapDisk(iSurf));
    side = 1;
    if (!finalTSOS.isValid()) {
      finalTSOS = propagator.propagate(tsos, PFGeometry::barrelBound(iSurf));
      side = 0;
    }
  } else {
    finalTSOS = propagator.propagate(tsos, PFGeometry::negativeEndcapDisk(iSurf));
    side = -1;
    if (!finalTSOS.isValid()) {
      finalTSOS = propagator.propagate(tsos, PFGeometry::barrelBound(iSurf));
      side = 0;
    }
  }

  if( !finalTSOS.isValid() ) {
    LogError("PFSimParticleProducer")<<"invalid trajectory state on surface: "
			  <<" iSurf = "<<iSurf
			  <<" tan theta = "<<p.perp()/p.z()
			  <<" pz = "<<p.z()
			  <<endl;
  }

  return finalTSOS;
}
