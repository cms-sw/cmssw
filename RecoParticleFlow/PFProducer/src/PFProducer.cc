#include "RecoParticleFlow/PFProducer/interface/PFProducer.h"

#include "RecoParticleFlow/PFAlgo/interface/PFBlock.h"
#include "RecoParticleFlow/PFAlgo/interface/PFBlockElement.h"
// #include "RecoParticleFlow/PFAlgo/interface/PFGeometry.h"

// #include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
// #include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
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
// #include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

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

PFProducer::PFProducer(const edm::ParameterSet& iConfig) :
  trackAlgo_(iConfig) {

  
  processRecTracks_ = 
    iConfig.getUntrackedParameter<bool>("process_RecTracks",true);

  processParticles_ = 
    iConfig.getUntrackedParameter<bool>("process_Particles",true);
  
  doParticleFlow_ = 
    iConfig.getUntrackedParameter<bool>("do_ParticleFlow",true);
    
  

  // use configuration file to setup input/output collection names
  recTrackModuleLabel_ 
    = iConfig.getUntrackedParameter<string>
    ("RecTrackModuleLabel","ckfTrackCandidates");

  pfClusterModuleLabel_ 
    = iConfig.getUntrackedParameter<string>
    ("PFClusterModuleLabel","particleFlowCluster");  

  pfClusterECALInstanceName_ 
    = iConfig.getUntrackedParameter<string>
    ("PFClusterECALInstanceName","ECAL");  

  pfClusterHCALInstanceName_ 
    = iConfig.getUntrackedParameter<string>
    ("PFClusterHCALInstanceName","HCAL");  

  pfClusterPSInstanceName_ 
    = iConfig.getUntrackedParameter<string>
    ("PFClusterPSInstanceName","PS");  

  simModuleLabel_ 
    = iConfig.getUntrackedParameter<string>
    ("SimModuleLabel","g4SimHits");



  // register products
  produces<reco::PFSimParticleCollection>();
  produces<reco::PFRecTrackCollection>();
  produces<reco::PFCandidateCollection>();
  // produces<reco::CandidateCollection>();
  

  // initialize track reconstruction ------------------------------
  fitterName_ = iConfig.getParameter<string>("Fitter");   
  propagatorName_ = iConfig.getParameter<string>("Propagator");
  builderName_ = iConfig.getParameter<string>("TTRHBuilder");   

  vertexGenerator_ = iConfig.getParameter<ParameterSet>
    ( "VertexGenerator" );   
  particleFilter_ = iConfig.getParameter<ParameterSet>
    ( "ParticleFilter" );   

  // initialize geometry parameters
  PFGeometry pfGeometry;
  
  
  // particle flow parameters  -----------------------------------
  pfReconMethod_ = iConfig.getParameter<int>("pf_recon_method");  

  string map_ECAL_eta 
    = iConfig.getParameter<string>("pf_resolution_map_ECAL_eta");  
  string map_ECAL_phi 
    = iConfig.getParameter<string>("pf_resolution_map_ECAL_phi");  
  //   will be necessary when preshower is used:
  //   string map_ECALec_x 
  //     = iConfig.getParameter<string>("pf_resolution_map_ECALec_x");  
  //   string map_ECALec_y 
  //     = iConfig.getParameter<string>("pf_resolution_map_ECALec_y");  
  string map_HCAL_eta 
    = iConfig.getParameter<string>("pf_resolution_map_HCAL_eta");  
  string map_HCAL_phi 
    = iConfig.getParameter<string>("pf_resolution_map_HCAL_phi");  

  
  try {
    PFBlock::setResMaps(map_ECAL_eta,
			map_ECAL_phi, 
			"",
			"",
			map_HCAL_eta,
			map_HCAL_phi);
  }
  catch( const string& err ) {
    LogError("PFProducer")<<" "<<err<<" -> PARTICLE FLOW DISABLED"<<endl;
    doParticleFlow_ = false;
  }
  

  double chi2_ECAL_HCAL 
    = iConfig.getParameter<double>("pf_chi2_ECAL_HCAL");  
  double chi2_ECAL_PS 
    = iConfig.getParameter<double>("pf_chi2_ECAL_PS");  
  double chi2_HCAL_PS 
    = iConfig.getParameter<double>("pf_chi2_HCAL_PS");  
  double chi2_ECAL_Track 
    = iConfig.getParameter<double>("pf_chi2_ECAL_Track");  
  double chi2_HCAL_Track 
    = iConfig.getParameter<double>("pf_chi2_HCAL_Track");  
  double chi2_PS_Track 
    = iConfig.getParameter<double>("pf_chi2_PS_Track");  

  PFBlock::setMaxChi2(chi2_ECAL_HCAL,
		      chi2_ECAL_PS,
		      chi2_HCAL_PS,
		      chi2_ECAL_Track,
		      chi2_HCAL_Track,
		      chi2_PS_Track );
  double nsigma 
    = iConfig.getParameter<double>("pf_nsigma_neutral");  
  PFBlock::setNsigmaNeutral(nsigma);

  double ecalibP0
    = iConfig.getParameter<double>("pf_ECAL_calib_p0");  
  double ecalibP1
    = iConfig.getParameter<double>("pf_ECAL_calib_p1");  
  PFBlock::setEcalib(ecalibP0, ecalibP1);
  
  // mySimEvent =  new FSimEvent(vertexGenerator_, particleFilter_);
  mySimEvent =  new FSimEvent( particleFilter_ );
}



PFProducer::~PFProducer() { delete mySimEvent; }


void 
PFProducer::beginJob(const edm::EventSetup & es)
{
  
  // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  es.getData(pdt);
  if ( !ParticleTable::instance() ) ParticleTable::instance(&(*pdt));
  mySimEvent->initializePdt(&(*pdt));
}


void PFProducer::produce(Event& iEvent, 
			 const EventSetup& iSetup) 
{
  
  LogDebug("PFProducer")<<"START event: "<<iEvent.id().event()
			<<" in run "<<iEvent.id().run()<<endl;
  
  
  set< PFBlockElement* > allElements; 
  
  // output collection for rectracks. will be used for particle flow
  auto_ptr< reco::PFRecTrackCollection > 
    pOutputPFRecTrackCollection(new reco::PFRecTrackCollection);
  auto_ptr< reco::PFSimParticleCollection > 
    pOutputPFSimParticleCollection(new reco::PFSimParticleCollection ); 
  auto_ptr< reco::PFCandidateCollection > 
    pOutputCandidateCollection(new reco::PFCandidateCollection ); 
//   auto_ptr< reco::CandidateCollection > 
//     pOutputCandidateCollection(new reco::CandidateCollection ); 
  

  // deal with RecTracks
  if(processRecTracks_) {
    
    processRecTracks(pOutputPFRecTrackCollection,
		     iEvent, iSetup);
        
    // iEvent.put(pOutputPFRecTrackCollection);
  }

  
  // deal with true particles 
  if( processParticles_) {
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
    mySimEvent->print();
//     cout<<"ntracks   = "<<mySimEvent->nTracks()<<endl;
//     cout<<"ngenparts = "<<mySimEvent->nGenParts()<<endl;

    const std::vector<FSimTrack>& fsimTracks = *(mySimEvent->tracks() );
    for(unsigned i=0; i<fsimTracks.size(); i++) {
    
      const FSimTrack& fst = fsimTracks[i];

      int motherId = -1;
      if( ! fst.noMother() ) 
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
	  pointEnd( 1, -1,
		    posEnd, momEnd);
	
	particle.addPoint(pointEnd);
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
      if( fst.onLayer2() ) { // PS layer2
	const RawParticle& rp = fst.layer2Entrance();
      
	math::XYZPoint posLayer2( rp.x(), rp.y(), rp.z() );
	math::XYZTLorentzVector momLayer2( rp.px(), rp.py(), rp.pz(), rp.e() );
	reco::PFTrajectoryPoint layer2Pt(-1, reco::PFTrajectoryPoint::PS2, 
					 posLayer2, momLayer2);
	
	particle.addPoint( layer2Pt ); 

	// extrapolate to cluster depth
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
          
      pOutputPFSimParticleCollection->push_back( particle );
    }

//     iEvent.put(pOutputPFSimParticleCollection);
  }
  
  
  if(doParticleFlow_) {
    
    LogDebug("PFProducer")<<"particle flow is starting"<<endl;
  
    // get ECAL, HCAL and PS clusters
    // add all clusters to set of particle flow elements

    Handle< vector<reco::PFCluster> > clustersECAL;
    try{      
      LogDebug("PFProducer")<<"get ECAL clusters"<<endl;
      iEvent.getByLabel(pfClusterModuleLabel_, pfClusterECALInstanceName_, 
			clustersECAL);

      for(unsigned i=0; i<clustersECAL->size(); i++) {
	if( (*clustersECAL)[i].type() != reco::PFCluster::TYPE_PF ) continue;
	
	reco::PFCluster *ncclust 
	  = const_cast<reco::PFCluster *> (& (*clustersECAL)[i] );
	
	allElements.insert( new PFBlockElementECAL( ncclust ) );
      }
      
    } catch (cms::Exception& err) { 
      LogError("PFProducer")<<err
			    <<" cannot get collection "
			    <<pfClusterModuleLabel_<<":"
			    <<pfClusterECALInstanceName_
			    <<endl;

    }
    
  
    Handle< vector<reco::PFCluster> > clustersHCAL;
    try{      
      LogDebug("PFProducer")<<"get HCAL clusters"<<endl;
      iEvent.getByLabel(pfClusterModuleLabel_, pfClusterHCALInstanceName_, 
			clustersHCAL);
      
      for(unsigned i=0; i<clustersHCAL->size(); i++) {
	if( (*clustersHCAL)[i].type() != reco::PFCluster::TYPE_PF ) continue;
	
	reco::PFCluster *ncclust 
	  = const_cast<reco::PFCluster *> (& (*clustersHCAL)[i] );

	allElements.insert( new PFBlockElementHCAL( ncclust ) );
      }
      
    } catch (cms::Exception& err) { 
      LogError("PFProducer")<<err
			    <<" cannot get collection "
			    <<pfClusterModuleLabel_<<":"
			    <<pfClusterHCALInstanceName_
			    <<endl;
      // throw err;
    }
    

    Handle< vector<reco::PFCluster> > clustersPS;
    try{      
      LogDebug("PFProducer")<<"get PS clusters"<<endl;
      iEvent.getByLabel(pfClusterModuleLabel_, pfClusterPSInstanceName_, 
			clustersPS);

      for(unsigned i=0; i<clustersPS->size(); i++) {
	if( (*clustersPS)[i].type() != reco::PFCluster::TYPE_PF ) continue;

	reco::PFCluster *ncclust 
	  = const_cast<reco::PFCluster *> (& (*clustersPS)[i] );

	allElements.insert( new PFBlockElementPS( ncclust ) );
     }
      
    } catch (cms::Exception& err) { 
      LogError("PFProducer")<<err
			    <<" cannot get collection "
			    <<pfClusterModuleLabel_<<":"
			    <<pfClusterPSInstanceName_
			    <<endl;
      // throw err;
    }
    
    
    // add all tracks to set of particle flow elements
    
    for(unsigned i=0; i<pOutputPFRecTrackCollection->size(); i++) {
      reco::PFRecTrack* track = & (*pOutputPFRecTrackCollection)[i];
      allElements.insert( new PFBlockElementTrack(track) );  
    }       


    PFBlock::setAllElements( allElements );
    vector< PFBlock > allPFBs;
    
    for(PFBlock::IT iele = allElements.begin(); 
	iele != allElements.end(); iele++) {
      
      if( (*iele)->block() ) continue; // already associated
      
      allPFBs.push_back( PFBlock() );
      allPFBs.back().associate( 0, *iele );
            
      int efbcolor = 1;
      allPFBs.back().finalize(efbcolor, pfReconMethod_); 
    }


    ostringstream  str;
    str<<"Reconstructed particles : "<<endl;
    
    for(unsigned iefb = 0; iefb<allPFBs.size(); iefb++) {
      
      switch(pfReconMethod_) {
      case 1:
	allPFBs[iefb].reconstructParticles1();
	break;
      case 2:
	allPFBs[iefb].reconstructParticles2();
	break;
      case 3:
	allPFBs[iefb].reconstructParticles3();
	break;
      default:
	break;
      }    
      LogDebug("PFProducer")<<(allPFBs[iefb])<<endl;

      // for each reconstructed particle, 
      // create a particle candidate
      std::vector< PFBlockParticle >& recparts = allPFBs[iefb].particles();
      for(unsigned ip=0; ip<recparts.size(); ip++) {
	char charge = static_cast<char> ( recparts[ip].charge() );
	const math::XYZTLorentzVector& mom = recparts[ip].momentum();
	
	int type = recparts[ip].type();
	
	reco::PFCandidate::ParticleType id 
	  = reco::PFCandidate::X;
	switch( type ) {
	case 211:
	case 11:   // charged hadron
	  id = reco::PFCandidate::h; 
	  break; 
	case 22:   // photon
	  id = reco::PFCandidate::gamma; 
	  break; 
	case 130:  // neutral hadron
	  id = reco::PFCandidate::h0; 
	  break; 
	}

	reco::PFCandidate* candidate 
	  = new reco::PFCandidate( charge, mom, id );
	pOutputCandidateCollection->push_back( *candidate ); 

// 	reco::PFCandidate* candidate 
// 	  = new reco::PFCandidate( charge, mom, id );
// 	pOutputCandidateCollection->push_back( candidate ); 

	str<<recparts[ip]<<endl;
      } 

    }
    LogInfo("PFProducer") << str.str()<<endl;

    LogDebug("PFProducer")<<"particle flow done"<<endl;
  }
   
  for(PFBlock::IT iele = allElements.begin(); 
      iele != allElements.end(); iele++) {
    delete *iele;
  }
  
  LogDebug("PFProducer")<<"Putting products in the event"<<endl;
  
  iEvent.put(pOutputPFRecTrackCollection);
  iEvent.put(pOutputPFSimParticleCollection);
  iEvent.put(pOutputCandidateCollection);

  LogDebug("PFProducer")<<"STOP event: "<<iEvent.id().event()
			<<" in run "<<iEvent.id().run()<<endl;
}


void 
PFProducer::processRecTracks(auto_ptr< reco::PFRecTrackCollection >& 
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
      LogDebug("PFProducer")<<"get tracker geometry"<<endl;
      iSetup.get<TrackerDigiGeometryRecord>().get(theG);

      LogDebug("PFProducer")<<"get magnetic field"<<endl;
      iSetup.get<IdealMagneticFieldRecord>().get(theMF);
      
      LogDebug("PFProducer")<<"get the trajectory fitter from the ES"<<endl;
      iSetup.get<TrackingComponentsRecord>().get(fitterName_, theFitter);
      
      LogDebug("PFProducer")<<"get the trajectory propagator from the ES"<<endl;
      iSetup.get<TrackingComponentsRecord>().get(propagatorName_, thePropagator);
      
      LogDebug("PFProducer")<<"get the TransientTrackingRecHitBuilder"<<endl;
      iSetup.get<TransientRecHitRecord>().get(builderName_, theBuilder);
    }
    catch( exception& err ) {
      LogError("PFProducer")
	<<"PFProducer::processRecTracks : exception: "
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
      LogDebug("PFProducer")<<"get the TrackCandidateCollection"
			    <<" from the event, source is " 
			    <<recTrackModuleLabel_ <<endl;
      Handle<TrackCandidateCollection> theTCCollection;
      iEvent.getByLabel(recTrackModuleLabel_, theTCCollection);

      //run the algorithm  
      LogDebug("PFProducer")<<"run the tracking algorithm"<<endl;
      trackAlgo_.runWithCandidate(theG.product(), theMF.product(), 
				  *theTCCollection,
				  theFitter.product(), thePropagator.product(),
				  theBuilder.product(), algoResults);
    } catch (cms::Exception& e) { 
      LogError("PFProducer")<<"cms::Exception caught : " 
			    << "cannot get collection " 
			    << recTrackModuleLabel_<<endl<<e<<endl;
    }

    // Loop over smoothed tracks and fill PFRecTrack collection 
    for(AlgoProductCollection::iterator itTrack = algoResults.begin();
	itTrack != algoResults.end(); itTrack++) {
      Trajectory*  theTraj  = (*itTrack).first;
      vector<TrajectoryMeasurement> measurements = theTraj->measurements();

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
      LogDebug("PFProducer")<<"closest approach point "<<closestPt<<endl;
    
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
					   reco::PFTrajectoryPoint::BeamPipe, 
					   posBeamPipe, momBeamPipe);
	
	track.addPoint(beamPipePt);
	LogDebug("PFProducer")<<"beam pipe point "<<beamPipePt<<endl;
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
	LogDebug("PFProducer")<<"add measuremnt "<<iTraj<<" "<<trajPt<<endl;
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
	  LogDebug("PFProducer")<<"ps1 point "<<ps1Pt<<endl;
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
	  LogDebug("PFProducer")<<"ps2 point "<<ps2Pt<<endl;
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
      LogDebug("PFProducer")<<"ecal point "<<ecalPt<<endl;

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
      LogDebug("PFProducer")<<"ecal shower maximum point "<<eSMaxPt 
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
	LogDebug("PFProducer")<<"hcal point "<<hcalPt<<endl;    

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
	// 	LogDebug("PFProducer")<<"hcal exit point "<<hcalExitPt<<endl;    
      }
      catch( exception& err) {
	LogError("PFProducer")<<"Exception : "<<err.what()<<endl;
	throw err; 
      }


      trackCollection->push_back(track);
      LogDebug("PFProducer")<<"PFRecTrack added to event"<<track<<endl;
    }   
}



TrajectoryStateOnSurface 
PFProducer::getStateOnSurface( PFGeometry::Surface_t iSurf, 
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
    LogError("PFProducer")<<"invalid trajectory state on surface: "
			  <<" iSurf = "<<iSurf
			  <<" tan theta = "<<p.perp()/p.z()
			  <<" pz = "<<p.z()
			  <<endl;
  }

  return finalTSOS;
}

//define this as a plug-in
// DEFINE_FWK_MODULE(PFProducer);
