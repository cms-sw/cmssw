#include "RecoParticleFlow/PFSimProducer/plugins/PFSimParticleProducer.h"

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
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
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

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"

#include <set>
#include <sstream>

using namespace std;
using namespace edm;

PFSimParticleProducer::PFSimParticleProducer(const edm::ParameterSet& iConfig) 
{


  processParticles_ = 
    iConfig.getUntrackedParameter<bool>("process_Particles",true);
    

  inputTagSim_ 
    = iConfig.getParameter<InputTag>("sim");

  //retrieving collections for MC Truth Matching

  //modif-beg
  inputTagFamosSimHits_ 
    = iConfig.getUntrackedParameter<InputTag>("famosSimHits");
  mctruthMatchingInfo_ = 
    iConfig.getUntrackedParameter<bool>("MCTruthMatchingInfo",false);
  //modif-end

  inputTagRecTracks_ 
    = iConfig.getParameter<InputTag>("RecTracks");
  inputTagEcalRecHitsEB_ 
    = iConfig.getParameter<InputTag>("ecalRecHitsEB");
  inputTagEcalRecHitsEE_ 
    = iConfig.getParameter<InputTag>("ecalRecHitsEE");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);


  // register products
  produces<reco::PFSimParticleCollection>();
  
  particleFilter_ = iConfig.getParameter<ParameterSet>
    ( "ParticleFilter" );   
  
  mySimEvent =  new FSimEvent( particleFilter_ );
}



PFSimParticleProducer::~PFSimParticleProducer() 
{ 
  delete mySimEvent; 
}


void 
PFSimParticleProducer::beginRun(const edm::Run& run,
				const edm::EventSetup & es)
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
 
  //MC Truth Matching only with Famos and UnFoldedMode option to true!!

  //vector to store the trackIDs of rectracks corresponding
  //to the simulated particle.
  std::vector<unsigned> recTrackSimID;
  
  //In order to know which simparticule contribute to 
  //a given Ecal RecHit energy, we need to access 
  //the PCAloHit from FastSim. 
  
  typedef std::pair<double, unsigned> hitSimID;
  typedef std::list< std::pair<double, unsigned> >::iterator ITM;
  std::vector< std::list <hitSimID> > caloHitsEBID(62000);
  std::vector<double> caloHitsEBTotE(62000,0.0);
  
  if(mctruthMatchingInfo_){
     
    //getting the PCAloHit
    edm::Handle<edm::PCaloHitContainer> pcalohits;
    //   bool found_phit 
    //     = iEvent.getByLabel("famosSimHits","EcalHitsEB",
    // 			pcalohits);  
    //modif-beg
    bool found_phit 
      = iEvent.getByLabel(inputTagFamosSimHits_,pcalohits);
    //modif-end
    
    if(!found_phit) {
      ostringstream err;
      err<<"could not find pcaloHit "<<"famosSimHits:EcalHitsEB";
      LogError("PFSimParticleProducer")<<err.str()<<endl;
      
      throw cms::Exception( "MissingProduct", err.str());
    }
    else {
      assert( pcalohits.isValid() );
      
      //     cout << "PFSimParticleProducer: number of pcalohits="
      // 	 << pcalohits->size() << endl;
      
      edm::PCaloHitContainer::const_iterator it    
	= pcalohits.product()->begin();
      edm::PCaloHitContainer::const_iterator itend 
	= pcalohits.product()->end();
      
      //loop on the PCaloHit from FastSim Calorimetry
      for(;it!=itend;++it)
	{
	  EBDetId detid(it->id());
	  
	  // 	cout << detid << " " << detid.rawId()
	  // 	     << " " <<  detid.hashedIndex() 
	  // 	     << " " << it->energy() 
	  // 	     << " " << it->id() 
	  // 	     << " trackId=" 
	  // 	     << it->geantTrackId() 
	  // 	     << endl;
	  
	  if(it->energy() > 0.0) {
	    std::pair<double, unsigned> phitsimid
	      = make_pair(it->energy(),it->geantTrackId());
	    caloHitsEBID[detid.hashedIndex()].push_back(phitsimid);
	    caloHitsEBTotE[detid.hashedIndex()] 
	      += it->energy(); //summing pcalhit energy	  
	  }//energy > 0
	  
	}//loop PcaloHits    
    }//pcalohit handle access
    
    //Retrieving the PFRecTrack collection for
    //Monte Carlo Truth Matching tool
    Handle< reco::PFRecTrackCollection > recTracks;
    try{      
      LogDebug("PFSimParticleProducer")<<"getting PFRecTracks"<<endl;
      iEvent.getByLabel(inputTagRecTracks_, recTracks);
      
    } catch (cms::Exception& err) { 
      LogError("PFSimParticleProducer")<<err
				       <<" cannot get collection "
				       <<"particleFlowBlock"<<":"
				       <<""
				       <<endl;
    }//pfrectrack handle access
    
    //getting the simID corresponding to 
    //each of the PFRecTracks
    getSimIDs( recTracks, recTrackSimID );
    
  }//mctruthMatchingInfo_ //modif
  
  // deal with true particles 
  if( processParticles_) {

    auto_ptr< reco::PFSimParticleCollection > 
      pOutputPFSimParticleCollection(new reco::PFSimParticleCollection ); 

    Handle<vector<SimTrack> > simTracks;
    bool found = iEvent.getByLabel(inputTagSim_,simTracks);
    if(!found) {

      ostringstream err;
      err<<"cannot find sim tracks: "<<inputTagSim_;
      LogError("PFSimParticleProducer")<<err.str()<<endl;
      
      throw cms::Exception( "MissingProduct", err.str());
    }
      
    
    
    Handle<vector<SimVertex> > simVertices;
    found = iEvent.getByLabel(inputTagSim_,simVertices);
    if(!found) {
      LogError("PFSimParticleProducer")
	<<"cannot find sim vertices: "<<inputTagSim_<<endl;
      return;
    }

      

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

      //This is finding out the simID corresponding 
      //to the recTrack
//       cout << "Particle " << i 
// 	   << " " << fst.genpartIndex() 
// 	   << " -------------------------------------" << endl;

      //GETTING THE TRACK ID
      unsigned         recTrackID = 99999;
      vector<unsigned> recHitContrib; //modif
      vector<double>   recHitContribFrac; //modif

      if(mctruthMatchingInfo_){ //modif

	for(unsigned lo=0; lo<recTrackSimID.size(); 
	    lo++) {
	  if( i == recTrackSimID[lo] ) {
// 	    cout << "Corresponding Rec Track " 
// 		 << lo << endl;
	    recTrackID = lo;
	  }//match track
	}//loop rectrack
// 	if( recTrackID == 99999 ) 
// 	  cout << "Sim Track not reconstructed pT=" <<  
// 	    fst.momentum().pt() << endl;
	
	// get the ecalBarrel rechits for MC truth matching tool
	edm::Handle<EcalRecHitCollection> rhcHandle;
	bool found = iEvent.getByLabel(inputTagEcalRecHitsEB_, 
				       rhcHandle);
	if(!found) {
	  ostringstream err;
	  err<<"could not find rechits "<< inputTagEcalRecHitsEB_;
	  LogError("PFSimParticleProducer")<<err.str()<<endl;
	  
	  throw cms::Exception( "MissingProduct", err.str());
	}
	else {
	  assert( rhcHandle.isValid() );
// 	  cout << "PFSimParticleProducer: number of rechits="
// 	       << rhcHandle->size() << endl;
	  
	  EBRecHitCollection::const_iterator it_rh    
	    = rhcHandle.product()->begin();
	  EBRecHitCollection::const_iterator itend_rh 
	    = rhcHandle.product()->end();
	  
	  for(;it_rh!=itend_rh;++it_rh)
	    {
	      unsigned rhit_hi 
		= EBDetId(it_rh->id()).hashedIndex();
	      EBDetId detid(it_rh->id());
// 	    cout << detid << " " << detid.rawId()
// 		 << " " <<  detid.hashedIndex() 
// 		 << " " << it_rh->energy() << endl;
	    
	      ITM it_phit    = caloHitsEBID[rhit_hi].begin();
	      ITM itend_phit = caloHitsEBID[rhit_hi].end();    
	      for(;it_phit!=itend_phit;++it_phit)
		{
		  if(i == it_phit->second)
		    {
		      //Alex (08/10/08) TO BE REMOVED, eliminating
		      //duplicated rechits
		      bool alreadyin = false;
		      for( unsigned ihit = 0; ihit < recHitContrib.size(); 
			   ++ihit )
			if(detid.rawId() == recHitContrib[ihit]) 
			  alreadyin = true;
		      
		      if(!alreadyin){		
			double pcalofraction = 0.0;
			if(caloHitsEBTotE[rhit_hi] != 0.0)
			  pcalofraction 
			    = (it_phit->first/caloHitsEBTotE[rhit_hi])*100.0;
			
			//store info
			recHitContrib.push_back(it_rh->id());
			recHitContribFrac.push_back(pcalofraction);
		      }//selected rechits	    
		    }//matching
		}//loop pcalohit
	      
	    }//loop rechits
	  
	}//getting the rechits

      }//mctruthMatchingInfo_ //modif

//       cout << "This particule has " << recHitContrib.size() 
// 	   << " rechit contribution" << endl;
//       for( unsigned ih = 0; ih < recHitContrib.size(); ++ih )
// 	cout << recHitContrib[ih] 
// 	     << " f=" << recHitContribFrac[ih] << " ";
//       cout << endl;

      reco::PFSimParticle particle(  fst.charge(), 
				     fst.type(), 
				     fst.id(), 
				     motherId,
				     fst.daughters(),
				     recTrackID,
				     recHitContrib,
				     recHitContribFrac);


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

	const RawParticle& rpout = fst.hcalExit();
	
	math::XYZPoint posHCALout( rpout.x(), rpout.y(), rpout.z() );
	math::XYZTLorentzVector momHCALout( rpout.px(), rpout.py(), rpout.pz(),
					    rpout.e() );
	reco::PFTrajectoryPoint 
	  hcalPtout(-1, reco::PFTrajectoryPoint::HCALExit, 
		     posHCALout, momHCALout);
	
	particle.addPoint( hcalPtout ); 	

	const RawParticle& rpho = fst.hoEntrance();
	
	math::XYZPoint posHOEntrance( rpho.x(), rpho.y(), rpho.z() );
	math::XYZTLorentzVector momHOEntrance( rpho.px(), rpho.py(), rpho.pz(),
					    rpho.e() );
	reco::PFTrajectoryPoint 
	  hoPtin(-1, reco::PFTrajectoryPoint::HOLayer, 
		 posHOEntrance, momHOEntrance);
	
	particle.addPoint( hoPtin ); 	




      }
      else { // add a dummy point
	reco::PFTrajectoryPoint dummy;
	particle.addPoint(dummy); 
      }
          
      pOutputPFSimParticleCollection->push_back( particle );
    }
    
    iEvent.put(pOutputPFSimParticleCollection);
  }

  
  LogDebug("PFSimParticleProducer")<<"STOP event: "<<iEvent.id().event()
				   <<" in run "<<iEvent.id().run()<<endl;
}

void PFSimParticleProducer::getSimIDs( const TrackHandle& trackh,
				       std::vector<unsigned>& recTrackSimID )
{

  if( trackh.isValid() ) {
//     cout << "Size=" << trackh->size() << endl;

    for(unsigned i=0;i<trackh->size(); i++) {
      
      reco::PFRecTrackRef ref( trackh,i );
      const reco::PFRecTrack& PFT   = *ref;
      const reco::TrackRef trackref = PFT.trackRef();

//       double Pt  = trackref->pt(); 
//       double DPt = trackref->ptError();
//       cout << " PFBlockProducer: PFrecTrack->Track Pt= " 
// 	   << Pt << " DPt = " << DPt << endl;
      
      trackingRecHit_iterator rhitbeg 
	= trackref->recHitsBegin();
      trackingRecHit_iterator rhitend 
	= trackref->recHitsEnd();
      for (trackingRecHit_iterator it = rhitbeg;  
	   it != rhitend; it++){

	if( it->get()->isValid() ){

	  const SiTrackerGSMatchedRecHit2D * rechit 
	    = (const SiTrackerGSMatchedRecHit2D*) (it->get());
	  
// 	  cout <<  "rechit" 	       
// 	       << " corresponding simId " 
// 	       << rechit->simtrackId() 
// 	       << endl;

	  recTrackSimID.push_back( rechit->simtrackId() );
	  break;
	}
      }//loop track rechit
    }//loop recTracks
  }//track handle valid
}
