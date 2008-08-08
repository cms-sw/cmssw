#include <iostream>
#include "ResidualRefitting.h"
#include <iomanip>

//framework includes
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

//#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// New Crazy idea

#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonProducer.h"
#include "RecoMuon/GlobalMuonProducer/src/TevMuonProducer.h"

using namespace std;
using namespace edm;
using namespace reco;


ResidualRefitting::ResidualRefitting( const ParameterSet & cfg ) :
  outputFileName_		( cfg.getUntrackedParameter<string>("histoutputFile") ),
  muons_     			( cfg.getParameter<InputTag>( "muons"		) ),
/*  muonsRemake_			( cfg.getParameter<InputTag>("muonsRemake"	) ),			//This Feels Misalignment
  muonsNoStation1_		( cfg.getParameter<InputTag>("muonsNoStation1") ),
  muonsNoStation2_		( cfg.getParameter<InputTag>("muonsNoStation2") ),
  muonsNoStation3_		( cfg.getParameter<InputTag>("muonsNoStation3") ),
  muonsNoStation4_		( cfg.getParameter<InputTag>("muonsNoStation4") ),
  muonsNoPXBLayer1_		( cfg.getParameter<InputTag>("muonsNoPXBLayer1"	) ),
  muonsNoPXBLayer2_		( cfg.getParameter<InputTag>("muonsNoPXBLayer1"	) ),
  muonsNoPXBLayer3_		( cfg.getParameter<InputTag>("muonsNoPXBLayer1"	) ),

  muonsNoTIBLayer1_			( cfg.getParameter<InputTag>("muonsNoTIBLayer1"	) ),
  muonsNoTIBLayer2_			( cfg.getParameter<InputTag>("muonsNoTIBLayer2"	) ),
  muonsNoTIBLayer3_			( cfg.getParameter<InputTag>("muonsNoTIBLayer3"	) ),
  muonsNoTIBLayer4_			( cfg.getParameter<InputTag>("muonsNoTIBLayer4"	) ),

  muonsNoTOBLayer1_			( cfg.getParameter<InputTag>("muonsNoTOBLayer1"	) ),
  muonsNoTOBLayer2_			( cfg.getParameter<InputTag>("muonsNoTOBLayer2"	) ),
  muonsNoTOBLayer3_			( cfg.getParameter<InputTag>("muonsNoTOBLayer3"	) ),
  muonsNoTOBLayer4_			( cfg.getParameter<InputTag>("muonsNoTOBLayer4"	) ),
  muonsNoTOBLayer5_			( cfg.getParameter<InputTag>("muonsNoTOBLayer5"	) ),
  muonsNoTOBLayer6_			( cfg.getParameter<InputTag>("muonsNoTOBLayer6"	) ),*/
  debug_				( cfg.getUntrackedParameter<bool>("doDebug"	) )
{
// service parameters
	edm::ParameterSet serviceParameters = cfg.getParameter<ParameterSet>("ServiceParameters");
  
// the services
	theService = new MuonServiceProxy(serviceParameters);
	
}  //The constructor

void ResidualRefitting::analyze(const Event& event, const EventSetup& eventSetup) {


	printf("STARTING EVENT\n");
	using namespace edm;

	// Generator Collection

// The original muon collection that is sitting in memory
	edm::Handle<MuonCollection> muons;
	event.getByLabel( muons_, muons ); //set label to muons

		

/*
// The new muoncollection from this fitting and misalingment scenario
	edm::Handle<MuonCollection> muonsRemakeColl;
	event.getByLabel(muonsRemake_, muonsRemakeColl);

//	std::cout<<"Muon Collection No Station 1"<<std::endl;
	edm::Handle<MuonCollection> muonsNoStation1Coll;
	event.getByLabel(muonsNoStation1_, muonsNoStation1Coll);
//	std::cout<<"Muon Collection No Station 2"<<std::endl;
	edm::Handle<MuonCollection> muonsNoStation2Coll;
	event.getByLabel(muonsNoStation2_, muonsNoStation2Coll);
//	std::cout<<"Muon Collection No Station 3"<<std::endl;
	edm::Handle<MuonCollection> muonsNoStation3Coll;
	event.getByLabel(muonsNoStation3_, muonsNoStation3Coll);	
//	std::cout<<"Muon Collection No Station 4"<<std::endl;
	edm::Handle<MuonCollection> muonsNoStation4Coll;
	event.getByLabel(muonsNoStation4_, muonsNoStation4Coll);

//	std::cout<<"Muon Collection No PXB "<<std::endl;
//Tracker Barrel Pixel Refits
	edm::Handle<MuonCollection> muonsNoPXBLayer1Coll;
	event.getByLabel(muonsNoPXBLayer1_, muonsNoPXBLayer1Coll);
	edm::Handle<MuonCollection> muonsNoPXBLayer2Coll;
	event.getByLabel(muonsNoPXBLayer2_, muonsNoPXBLayer2Coll);
	edm::Handle<MuonCollection> muonsNoPXBLayer3Coll;
	event.getByLabel(muonsNoPXBLayer3_, muonsNoPXBLayer3Coll);
//	std::cout<<"Muon Collection No TIB "<<std::endl;
// Tracker Inner Barrel Refits
	edm::Handle<MuonCollection> muonsNoTIBLayer1Coll;
	event.getByLabel(muonsNoTIBLayer1_, muonsNoTIBLayer1Coll);
	edm::Handle<MuonCollection> muonsNoTIBLayer2Coll;
	event.getByLabel(muonsNoTIBLayer2_, muonsNoTIBLayer2Coll);
	edm::Handle<MuonCollection> muonsNoTIBLayer3Coll;
	event.getByLabel(muonsNoTIBLayer3_, muonsNoTIBLayer3Coll);
	edm::Handle<MuonCollection> muonsNoTIBLayer4Coll;
	event.getByLabel(muonsNoTIBLayer4_, muonsNoTIBLayer4Coll);

//	std::cout<<"Muon Collection No TOB "<<std::endl;

//Tracker outer barrel refits
	edm::Handle<MuonCollection> muonsNoTOBLayer1Coll;
	event.getByLabel(muonsNoTOBLayer1_, muonsNoTOBLayer1Coll);
	edm::Handle<MuonCollection> muonsNoTOBLayer2Coll;
	event.getByLabel(muonsNoTOBLayer2_, muonsNoTOBLayer2Coll);
	edm::Handle<MuonCollection> muonsNoTOBLayer3Coll;
	event.getByLabel(muonsNoTOBLayer3_, muonsNoTOBLayer3Coll);
	edm::Handle<MuonCollection> muonsNoTOBLayer4Coll;
	event.getByLabel(muonsNoTOBLayer4_, muonsNoTOBLayer4Coll);
	edm::Handle<MuonCollection> muonsNoTOBLayer5Coll;
	event.getByLabel(muonsNoTOBLayer5_, muonsNoTOBLayer5Coll);
	edm::Handle<MuonCollection> muonsNoTOBLayer6Coll;
	event.getByLabel(muonsNoTOBLayer6_, muonsNoTOBLayer6Coll);
 
	//magnetic field information	
	edm::ESHandle<MagneticField> field;
	eventSetup.get<IdealMagneticFieldRecord>().get(field);
	theField = &*field;
	theService->update(eventSetup);

//	std::cout<<"Going to zero storage"<<std::endl;
    zero_storage();
  */
//Do the Gmr Muons from the unModified Collection
	int iGmr = 0;
	for ( MuonCollection::const_iterator muon = muons->begin(); muon!=muons->end(); muon++, iGmr++) {

		std::cout<<iGmr<<std::endl;
		
//		if (debug_) {
			printf("Data Dump:: Original GMR Muons\n");
			dumpRecoMuonColl(muon);		
//		}
	}
	
/*****************************************************************************************************************/
/*
 * Do the Gmr Muons from the Rebuild collection 
 * This holds	1) p, pt, eta, phi
 *				2) info on sam rec hits
 *				3) Extrapolate to rec hits from GMR track with Rho and Z -- good for DTs and CSCs
*/
/*
//Store the basic Muon information
	if (debug_) std::cout<<"Global Muon sensitive to misalignment"<<std::endl;
	int iGmrRemake = 0;
	for ( MuonCollection::const_iterator muon = muonsRemakeColl->begin(); muon!=muonsRemakeColl->end(); muon++, iGmrRemake++) {
		if ( iGmrRemake >= ResidualRefitting::N_MAX_STORED) break; // error checking

		if (debug_) {
			printf("Data Dump:: Rebuilt GMR Muon Collection\n");
			dumpRecoMuonColl(muon);		
		}
		
		muonInfo(storageGmrNew_, muon->combinedMuon()	, iGmrRemake);	//Store combined global muon
		muonInfo(storageSamNew_, muon->standAloneMuon()	, iGmrRemake);	//Store stand alone muon
		muonInfo(storageTrkNew_, muon->track()			, iGmrRemake);	//Store tracker muon
	}
	storageGmrNew_.n_			= iGmrRemake;
	storageSamNew_.n_			= iGmrRemake;
	storageTrkNew_.n_			= iGmrRemake;

	ResidualRefitting::collectMuonRecHits(muonsRemakeColl, storageRecMuon_, storageTrackExtrapRec_);
	ResidualRefitting::collectTrackerRecHits(muonsRemakeColl, storageTrackHit_, storageTrackExtrapTracker_);
//Muon Rec Hits

	
	omitStation(muonsNoStation1Coll, storageGmrNoSt1_, storageSamNoSt1_, storageTrackExtrapRecNoSt1_, 1);
	omitStation(muonsNoStation2Coll, storageGmrNoSt2_, storageSamNoSt2_, storageTrackExtrapRecNoSt2_, 2);
	omitStation(muonsNoStation3Coll, storageGmrNoSt3_, storageSamNoSt3_, storageTrackExtrapRecNoSt3_, 3);
	omitStation(muonsNoStation4Coll, storageGmrNoSt4_, storageSamNoSt4_, storageTrackExtrapRecNoSt4_, 4);																		
	omitTrackerSystem(muonsNoPXBLayer1Coll, storageGmrNoPXBLayer1, storageTrkNoPXBLayer1, storageTrackNoPXBLayer1, ResidualRefitting::PXB);
	omitTrackerSystem(muonsNoPXBLayer2Coll, storageGmrNoPXBLayer2, storageTrkNoPXBLayer2, storageTrackNoPXBLayer2, ResidualRefitting::PXB);
	omitTrackerSystem(muonsNoPXBLayer3Coll, storageGmrNoPXBLayer3, storageTrkNoPXBLayer3, storageTrackNoPXBLayer3, ResidualRefitting::PXB);
		
	omitTrackerSystem(muonsNoTIBLayer1Coll, storageGmrNoTIBLayer1, storageTrkNoTIBLayer1, storageTrackNoTIBLayer1, ResidualRefitting::TIB);
	omitTrackerSystem(muonsNoTIBLayer2Coll, storageGmrNoTIBLayer2, storageTrkNoTIBLayer2, storageTrackNoTIBLayer2, ResidualRefitting::TIB);
	omitTrackerSystem(muonsNoTIBLayer3Coll, storageGmrNoTIBLayer3, storageTrkNoTIBLayer3, storageTrackNoTIBLayer3, ResidualRefitting::TIB);
	omitTrackerSystem(muonsNoTIBLayer4Coll, storageGmrNoTIBLayer4, storageTrkNoTIBLayer4, storageTrackNoTIBLayer4, ResidualRefitting::TIB);
		
	omitTrackerSystem(muonsNoTOBLayer1Coll, storageGmrNoTOBLayer1, storageTrkNoTOBLayer1, storageTrackNoTOBLayer1, ResidualRefitting::TOB);
	omitTrackerSystem(muonsNoTOBLayer2Coll, storageGmrNoTOBLayer2, storageTrkNoTOBLayer2, storageTrackNoTOBLayer2, ResidualRefitting::TOB);
	omitTrackerSystem(muonsNoTOBLayer3Coll, storageGmrNoTOBLayer3, storageTrkNoTOBLayer3, storageTrackNoTOBLayer3, ResidualRefitting::TOB);
	omitTrackerSystem(muonsNoTOBLayer4Coll, storageGmrNoTOBLayer4, storageTrkNoTOBLayer4, storageTrackNoTOBLayer4, ResidualRefitting::TOB);
	omitTrackerSystem(muonsNoTOBLayer5Coll, storageGmrNoTOBLayer5, storageTrkNoTOBLayer5, storageTrackNoTOBLayer5, ResidualRefitting::TOB);
	omitTrackerSystem(muonsNoTOBLayer6Coll, storageGmrNoTOBLayer6, storageTrkNoTOBLayer6, storageTrackNoTOBLayer6, ResidualRefitting::TOB);

	
	dumpTrackHits(storageTrackHit_);


//	omitTrackerSystem(muonsNoPXFColl, storageGmrNoPXF, storageTrkNoPXF, storageTrackNoPXF, ResidualRefitting::PXF);
//	omitTrackerSystem(muonsNoTIDColl, storageGmrNoTID, storageTrkNoTID, storageTrackNoTID, ResidualRefitting::TID);
//	omitTrackerSystem(muonsNoTECColl, storageGmrNoTEC, storageTrkNoTEC, storageTrackNoTEC, ResidualRefitting::TEC);

*/
/*  
	dumpTrackExtrap(storageTrackExtrapRecNoSt1_);
	dumpTrackExtrap(storageTrackExtrapRecNoSt2_);
	dumpTrackExtrap(storageTrackExtrapRecNoSt3_);
	dumpTrackExtrap(storageTrackExtrapRecNoSt4_);
	dumpTrackExtrap(storageTrackExtrapTracker_);

	dumpTrackExtrap(storageTrackNoPXBLayer1);
	dumpTrackHits(storageTrackHit_);
//	dumpMuonRecHits(storageRecMuon_);
*/	
/****************************************************************************************************************************************/

  
/*
 *	This is a bastardization of Ivan's code that extrapolates to a cylinder.
 *
*/

/*
	int iGmrCyl = 0;
	for (reco::MuonCollection::const_iterator muon = muonsRemakeColl->begin(); muon != muonsRemakeColl->end(); muon++, iGmrCyl++) {

		ResidualRefitting::cylExtrapTrkSam(iGmrCyl, muon->standAloneMuon()	, samExtrap120_		, 120.);
		ResidualRefitting::cylExtrapTrkSam(iGmrCyl, muon->track()			, trackExtrap120_	, 120.);

	}
	samExtrap120_.n_	 = iGmrCyl;
	trackExtrap120_.n_	 = iGmrCyl; 

	outputTree_ -> Fill();
  	std::cout << "FILLING NTUPLE!" << std::endl;

	std::cout << "Entries Recorded: " << outputTree_ -> GetEntries() << " Branch :: " << outputBranch_ -> GetEntries() <<		 std::endl<<std::endl;
	
*/
//  /*************************************************************************************************************/
//  //END OF ANALYSIS
//  //END OF ANALYSIS
//  /***********************************************************************************************************/
}
//end Analyze() main function

//------------------------------------------------------------------------------
//  
// Destructor
// 
ResidualRefitting::~ResidualRefitting() {
  delete outputFile_;
}
// 
// Store the Muon information for the Muon Rec Hits / refit with All stations THIS IS THE ONLY PLACE WHERE I STORE REC HITS
// 
void ResidualRefitting::collectMuonRecHits(edm::Handle<reco::MuonCollection> muonColl, ResidualRefitting::storage_hit& storeHit, ResidualRefitting::storage_trackExtrap& trackExtrap) {
	
	int iGmrRemake = 0;
	int iRecRemake = 0;
	int iExtrap = 0;
	if(debug_) printf("\nRemake Original GMR Muons as Collection\n");
	for ( MuonCollection::const_iterator muon = muonColl->begin(); muon!=muonColl->end(); muon++, iGmrRemake++) {
		if ( iExtrap >= ResidualRefitting::N_MAX_STORED_HIT) break; // error checking


// collect some information for track extrapolation
		SteppingHelixPropagator inwardPropRec  ( theField, oppositeToMomentum );
		SteppingHelixPropagator outwardPropRec ( theField, alongMomentum );
	
		FreeTrajectoryState recoStart = ResidualRefitting::freeTrajStateMuon(muon->combinedMuon());
					
//Begin the loop over the muon system rec hits	
		if (debug_) printf ("Looping over SAM rec hits in the original Global Muon...\n");
		for(trackingRecHit_iterator rec =  muon->standAloneMuon()->recHitsBegin();
		 rec != muon->standAloneMuon()->recHitsEnd(); rec++, iRecRemake++) {


	//Get Muon System Information		
			DetId detid = (*rec)->geographicalId(); 
			if (detid.det() != DetId::Muon) {
				std::cout<<"LOLZ! Not teh muon system"<<std::endl;
				continue;
			}
			int systemMuon  = detid.subdetId(); // 1 DT; 2 CSC; 3 RPC
			int endcap		= -999;
			int station		= -999;
			int ring		= -999;
			int chamber		= -999;
			int layer		= -999;
			int superLayer  = -999;
			double lpX		= -999;
			double lpY		= -999;
			double lpZ		= -999;
			if ( systemMuon == MuonSubdetId::CSC) {
				CSCDetId id(detid.rawId());
				endcap		= id.endcap();
				station		= id.station();
				ring		= id.ring();
				chamber		= id.chamber();
				layer		= id.layer();
				if (debug_)printf("%d System: CSC\n [endcap][station][ringN][chamber][layer]:[%d][%d][%d][%d][%d]\t",iRecRemake, endcap, station, ring, chamber, layer);

			}
			else if ( systemMuon == MuonSubdetId::DT ) {
				DTWireId id(detid.rawId());
				station		= id.station();
				layer		= id.layer();
				superLayer	= id.superLayer();
				if (debug_) std::cout<<iRecRemake<<" System: DT"<<std::endl;
				
			}
			else if ( systemMuon == MuonSubdetId::RPC) {
				RPCDetId id(detid.rawId());
				station		= id.station();
				if (debug_) std::cout<<iRecRemake<<"System: RPC"<<std::endl;
			}
			else printf("%d THIS ISN'T EVEN A MUON!!!!\n", iRecRemake);
// Local Coordinates
			LocalPoint lp = (*rec)->localPosition();
			lpX	= lp.x();
			lpY = lp.y();
			lpZ = lp.z();

// Global Coordinates		
			MuonTransientTrackingRecHit::MuonRecHitPointer mrhp =	
				MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((**rec).geographicalId())
				,&(**rec)); 

			GlobalPoint gp = mrhp->globalPosition();
			double gpRecX = gp.x();
			double gpRecY = gp.y();
			double gpRecZ = gp.z();
			double gpRecEta = gp.eta();
			double gpRecPhi = gp.phi();
			
						
			
			if (debug_) printf("Global Position\t\t\t x = %0.6f\t y = %0.6f\t z = %0.6f\n", gpRecX, gpRecY, gpRecZ);	
//Store 
			storeHit.muonLink_	[iRecRemake]	= iGmrRemake;			
			storeHit.system_	[iRecRemake]	= systemMuon;
			storeHit.endcap_	[iRecRemake]	= endcap;
			storeHit.station_	[iRecRemake]	= station;
			storeHit.ring_		[iRecRemake]	= ring;
			storeHit.chamber_	[iRecRemake]	= chamber;
			storeHit.layer_		[iRecRemake]	= layer;
			storeHit.superLayer_[iRecRemake]	= superLayer;

			storeHit.gpX_		[iRecRemake]	= gpRecX;
			storeHit.gpY_		[iRecRemake]	= gpRecY;
			storeHit.gpZ_		[iRecRemake]	= gpRecZ;
			storeHit.gpEta_		[iRecRemake]	= gpRecEta;
			storeHit.gpPhi_		[iRecRemake]	= gpRecPhi;
			storeHit.lpX_		[iRecRemake]	= lpX;
			storeHit.lpY_		[iRecRemake]	= lpY;
			storeHit.lpZ_		[iRecRemake]	= lpZ;


			int pars[3] = {iExtrap, iGmrRemake, iRecRemake};
			trkExtrap(trackExtrap, detid, pars, outwardPropRec, recoStart) ;
			iExtrap++;
			if ( iExtrap >= ResidualRefitting::N_MAX_STORED_HIT) {
				std::cout << " TOO Rec Hits ONLY FIRST " << ResidualRefitting::N_MAX_STORED_HIT << " WILL BE STORED! " << std::endl;      
				break;
			} 

		}

	}
	storeHit.n_		= iRecRemake;
	trackExtrap.n_	= iExtrap;
}
//
// Store the Muon information for the Tracker Rec hits / refit with all tracker systems THIS IS THE ONLY PLACE WHERE THE TRACKER INFO IS STORED
//
void ResidualRefitting::collectTrackerRecHits(edm::Handle<reco::MuonCollection> muonColl, ResidualRefitting::storage_trackHit& storeHit, ResidualRefitting::storage_trackExtrap& storeExtrap) {

	int iGmr = 0; 
	int iRec = 0;
	int iExtrap = 0;
	bool dump_ = debug_;
	if (dump_) std::cout<<"In the collectTrackerRecHits function\n";
	
	for ( MuonCollection::const_iterator muon = muonColl->begin(); muon != muonColl->end(); muon++, iGmr++) {
	
		if (debug_) dumpRecoMuonColl(muon); // dump the Gmr, Trk, and Sam  P, Pt, Eta, Phi

		SteppingHelixPropagator inwardProp  ( theField, oppositeToMomentum	);
		SteppingHelixPropagator outwardProp ( theField, alongMomentum		);
		FreeTrajectoryState recoStart = freeTrajStateMuon(muon->combinedMuon());

		
		int subDetectorLast = -1;
		for(trackingRecHit_iterator rec =  muon->track()->recHitsBegin(); 
		 rec != muon->track()->recHitsEnd(); rec++) {
			
			if (!(*rec)->isValid() ) continue;
			DetId detid = (*rec)->geographicalId(); 

			int detector	= -1;
			int subdetector = -1;
			int blade		= -1;
			int disk		= -1;
			int ladder		= -1;
			int layer		= -1;
			int module		= -1;
			int panel		= -1;
			int ring		= -1;
			int side		= -1;
			int wheel		= -1;
			
			
			double gpX		= -99999;
			double gpY		= -99999;
			double gpZ		= -99999;
			double gpEta	= -99999;
			double gpPhi	= -99999;
			double lpX		= -99999;
			double lpY		= -99999;
			double lpZ		= -99999;
			
//Detector Info

			detector = detid.det();
			subdetector = detid.subdetId();
			
			if (subdetector != subDetectorLast) {
				subDetectorLast = subdetector;
				if (dump_) std::cout<<std::endl;
			}
			
			if (detector != DetId::Tracker) { 
				std::cout<<"OMFG NOT THE TRACKER\n"<<std::endl;
				continue;
			}

			if (dump_) std::cout<<"Tracker:: ";
			if (subdetector == ResidualRefitting::PXB) {
				PXBDetId id(detid.rawId());
				layer	= id.layer();
				ladder	= id.ladder();
				module	= id.module();
				if (dump_)	std::cout	<<	"PXB"
										<<	"\tlayer = "	<< layer
										<<	"\tladder = "	<< ladder
										<<	"\tmodule = "	<< module;
				
			} 
			else if (subdetector == ResidualRefitting::PXF) {
				PXFDetId id(detid.rawId());
				side	= id.side();
				disk	= id.disk();
				blade	= id.blade();
				panel	= id.panel();
				module	= id.module();
				if (dump_)	std::cout	<<  "PXF"
										<<	"\tside = "		<< side
										<<	"\tdisk = "		<< disk
										<<	"\tblade = "	<< blade
										<<	"\tpanel = "	<< panel
										<<	"\tmodule = "	<< module;
							
			}
			else if (subdetector == ResidualRefitting::TIB) {
				TIBDetId id(detid.rawId());
				layer	= id.layer();
				module	= id.module();
				if (dump_)	std::cout	<< "TIB"
										<< "\tlayer = "	<< layer
										<< "\tmodule = "<< module;
			}
			else if (subdetector == ResidualRefitting::TID) {
				TIDDetId id(detid.rawId());
				side	= id.side();
				wheel	= id.wheel();
				ring	= id.ring();
				if (dump_)	std::cout	<<"TID"
										<< "\tside = "	<< side
										<< "\twheel = "	<< wheel
										<< "\tring = "	<< ring;
			
			}
			else if (subdetector == ResidualRefitting::TOB) {
				TOBDetId id(detid.rawId());
				layer	= id.layer();
				module	= id.module();
				if (dump_)	std::cout	<<"TOB"
										<<"\tlayer = "	<< layer
										<<"\tmodule = "	<< module;
			
			}
			else if (subdetector == ResidualRefitting::TEC) {
				TECDetId id(detid.rawId());
				ring	= id.ring();
				module	= id.module();
				if (dump_)	std::cout	<<"TEC"
										<< "\tring = "	<< ring
										<< "\tmodule = "<< module;
			}

//Global Point Info		
			MuonTransientTrackingRecHit::MuonRecHitPointer mrhp =	
			 MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((**rec).geographicalId()), &(**rec)); 			
				GlobalPoint gp = mrhp->globalPosition();							
				gpX	= gp.x();
				gpY	= gp.y();
				gpZ	= gp.z();
				gpEta = gp.eta();
				gpPhi = gp.phi();
				LocalPoint lp	= (*rec)->localPosition();
				lpX = lp.x();
				lpY = lp.y();
				lpZ = lp.z();
						
			if (debug_)	std::cout	<< setprecision(2)
//									<<  "Tracker Rec Hits: "
									<<	"\tX = "		<< gpX
//									<<	"\tY = "		<< gpY
									<<	"\tZ = "		<< gpZ
									<<  " \trho = "		<< sqrt( gpX * gpX + gpY * gpY )
//									<<	"\tEta = "		<< gpEta
//									<<	"\tPhi = "		<< gpPhi
									<<std::endl;
///////
//
///////
					
			storeHit.muonLink_		[iRec] = iGmr		;
			storeHit.detector_		[iRec] =detector	;
			storeHit.subdetector_	[iRec] =subdetector ;
			storeHit.blade_			[iRec] =blade		;
			storeHit.disk_			[iRec] =disk		;
			storeHit.ladder_		[iRec] =ladder		;
			storeHit.layer_			[iRec] =layer		;
			storeHit.module_		[iRec] =module		;
			storeHit.panel_			[iRec] =panel		;
			storeHit.ring_			[iRec] =ring		;
			storeHit.side_			[iRec] =side		;
			storeHit.wheel_			[iRec] =wheel		;
					
			storeHit.gpX_			[iRec] = gpX;
			storeHit.gpY_			[iRec] = gpY;
			storeHit.gpZ_			[iRec] = gpZ;
			storeHit.gpEta_			[iRec] = gpEta;
			storeHit.gpPhi_			[iRec] = gpPhi;
			storeHit.lpX_			[iRec] = lpX;
			storeHit.lpY_			[iRec] = lpY;
			storeHit.lpZ_			[iRec] = lpZ;

			int pars[3] = {iExtrap, iGmr, iRec};

			ResidualRefitting::trkExtrap(storeExtrap, detid, pars, outwardProp, recoStart);
			if (dump_)	{
				double xxTemp = storeExtrap.gpX_[iExtrap];
				double yyTemp = storeExtrap.gpY_[iExtrap];
				double zzTemp = storeExtrap.gpZ_[iExtrap];
				
				double rho1 = sqrt(gpX * gpX + gpY * gpY);
				double rho2 = sqrt(xxTemp * xxTemp + yyTemp * yyTemp);
						
						
				printf("\n\t\tRec Hits:\t z = %8.4f rho = %8.4f\n",gpZ, rho1 );
				printf("\t\tExtrap:\t\t z = %8.4f rho = %8.4f\n\n",zzTemp, rho2 );
			}
									
			iExtrap++;			
			iRec++;
			if ( iRec >= ResidualRefitting::N_MAX_STORED_HIT) { 
				std::cout << "Too many rec hits... Give up while you still can!"<<std::endl;
				break;
			}
		}				
	}
	storeHit.n_		= iRec;
	storeExtrap.n_	= iExtrap;


}
//
// Store Muon info on P, Pt, eta, phi
//
void ResidualRefitting::muonInfo(ResidualRefitting::storage_muon& storeMuon, reco::TrackRef muon, int val) {

		storeMuon.pt_ [val]		= muon->pt();
        storeMuon.p_  [val]		= muon->p();
        storeMuon.eta_[val]		= muon->eta();
		storeMuon.phi_[val]		= muon->phi();
		storeMuon.charge_[val]	= muon->charge();
}
//
// Run code to get store muon information and omit startion infor for the muons
// 
void ResidualRefitting::omitStation(edm::Handle<reco::MuonCollection> funcMuons, ResidualRefitting::storage_muon& storeGmr, ResidualRefitting::storage_muon& storeSam,
						ResidualRefitting::storage_trackExtrap& storeExtrap, int omitStation) {

	if (debug_) std::cout<<"Global Muon sensitive to misalignment : no station "<<omitStation<<std::endl;

	int iGmr = 0;
	int iRec = 0;
	int iExtrap = 0;
	if(debug_) printf("\n Original GMR Muons as Collection\n");
	for ( MuonCollection::const_iterator muon = funcMuons->begin(); muon!=funcMuons->end(); muon++, iGmr++) {
		if ( iRec >= ResidualRefitting::N_MAX_STORED_HIT) break; // error checking
		
//Store Muon Information		
		ResidualRefitting::muonInfo(storeGmr, muon->combinedMuon(), iGmr);
		ResidualRefitting::muonInfo(storeSam, muon->standAloneMuon(), iGmr);

//Helix propagator
		SteppingHelixPropagator inwardPropRec  ( theField, oppositeToMomentum );
		SteppingHelixPropagator outwardPropRec ( theField, alongMomentum );

		FreeTrajectoryState recoStart = freeTrajStateMuon(muon->combinedMuon());//( innerPoint, innerVec, muon ->charge(), theField ); 
	
//Begin the loop over the muon system rec hits	
		if (debug_) printf ("Looping over SAM rec hits in the original Global Muon...\n");
		for(trackingRecHit_iterator rec =  muon->standAloneMuon()->recHitsBegin();
		 rec != muon->standAloneMuon()->recHitsEnd(); rec++, iRec++) {


	//Get Muon System Information		
			DetId detid = (*rec)->geographicalId(); 
			if (detid.det() != DetId::Muon) {
				std::cout<<"OMFG Not a muon!\n"<<std::endl;
				continue;
			}
			int systemMuon = detid.subdetId(); // 1 DT; 2 CSC; 3 RPC
			int endcap		= -999;
			int station		= -999;
			int ring		= -999;
			int chamber		= -999;
			int layer		= -999;		
			if ( systemMuon == MuonSubdetId::CSC) {
				CSCDetId id(detid.rawId());
				endcap		= id.endcap();
				station		= id.station();
				ring		= id.ring();
				chamber		= id.chamber();
				layer		= id.layer();
				if (debug_)printf("%d System: CSC\t [endcap][station][ringN][chamber][layer]:[%d][%d][%d][%d][%d]\t",iRec, endcap, station, ring, chamber, layer);

			}
			else if ( systemMuon == MuonSubdetId::DT ) {
				DTWireId id(detid.rawId());
				station		= id.station();
				if (debug_) std::cout<<iRec<<" System: DT"<<std::endl;
				
			}
			else if ( systemMuon == MuonSubdetId::RPC) {
				RPCDetId id(detid.rawId());
				station		= id.station();
				if (debug_) std::cout<<iRec<<"System: RPC"<<std::endl;
			}
			else printf("%d THIS ISN'T EVEN A MUON!!!!\n", iRec);
			if (station != omitStation) continue;
// Global Coordinates		
			MuonTransientTrackingRecHit::MuonRecHitPointer mrhp =	
				MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((**rec).geographicalId())
				,&(**rec)); 

			GlobalPoint gp = mrhp->globalPosition();
			double gpRecX = gp.x();
			double gpRecY = gp.y();
			double gpRecZ = gp.z();			
//			double gpRecRho = sqrt ( gpRecX * gpRecX + gpRecY * gpRecY);
			
			
			if (debug_) printf("Global Position\t\t\t x = %0.6f\t y = %0.6f\t z = %0.6f\n", gpRecX, gpRecY, gpRecZ);	

			double gpExtrapX	= -99999;
			double gpExtrapY	= -99999;
			double gpExtrapZ	= -99999;
			double gpExtrapEta	= -99999;
			double gpExtrapPhi	= -99999;
			double lpX			= -99999;
			double lpY			= -99999;
			const GeomDet* gdet = theService->trackingGeometry()->idToDet(detid);
			
			TrajectoryStateOnSurface surfTest =  outwardPropRec.propagate(recoStart, gdet->surface());
			
			if (surfTest.isValid()) {
			
				GlobalPoint globTest	= surfTest.globalPosition();		
				gpExtrapX				= globTest.x();
				gpExtrapY				= globTest.y();
				gpExtrapZ				= globTest.z();
				gpExtrapEta				= globTest.eta();
				gpExtrapPhi				= globTest.phi();
				lpX			= surfTest.localPosition().x();
				lpY			= surfTest.localPosition().y();
			}

			if (debug_) std::cout	<<	"\t\t\t\t\tx = "<< gpExtrapX	
									<<	"\ty = "		<< gpExtrapY	
									<<	"\tz = "		<< gpExtrapZ	
									<<	"\t phi = "		<< gpExtrapPhi	
									<<	std::endl;
			storeExtrap.muonLink_	[iExtrap] = iGmr	;
			storeExtrap.recLink_	[iExtrap] = iRec	;
			storeExtrap.gpX_		[iExtrap] = gpExtrapX	;
			storeExtrap.gpY_		[iExtrap] = gpExtrapY	;
			storeExtrap.gpZ_		[iExtrap] = gpExtrapZ	;
			storeExtrap.gpEta_		[iExtrap] = gpExtrapEta	;
			storeExtrap.gpPhi_		[iExtrap] = gpExtrapPhi	;	
			storeExtrap.lpX_		[iExtrap] = lpX;
			storeExtrap.lpY_		[iExtrap] = lpY;


			iExtrap++;			
			if ( iExtrap >= ResidualRefitting::N_MAX_STORED_HIT) {
				std::cout << " TOO Rec Hits ONLY FIRST " << ResidualRefitting::N_MAX_STORED_HIT << " WILL BE STORED! " << std::endl;      
				break;
			} 
			if (debug_)std::cout << iGmr  << "\t" << iRec <<"\t" << iExtrap<<std::endl;
			
			
		}
	}
	storeGmr.n_ = iGmr;
	storeSam.n_ = iGmr;
	storeExtrap.n_ = iExtrap;

}
//
// Go through the tracker system and calculate GMR extrapolation values for GMRs that are in the omitted station
//
void ResidualRefitting::omitTrackerSystem(edm::Handle<reco::MuonCollection> trkMuons, ResidualRefitting::storage_muon& storeGmr, ResidualRefitting::storage_muon& storeTrk,
						ResidualRefitting::storage_trackExtrap& storeExtrap, int omitSystem) {

	int iGmr = 0; 
	int iRec = 0;
	int iExtrap = 0;
	bool dump_ = debug_;
	if (dump_) std::cout<<"In the omitTrackerSystem function\n";
	
	
	for ( MuonCollection::const_iterator muon = trkMuons->begin(); muon != trkMuons->end(); muon++, iGmr++) {
	
		if (debug_) dumpRecoMuonColl(muon); // dump the Gmr, Trk, and Sam  P, Pt, Eta, Phi
		ResidualRefitting::muonInfo( storeGmr, muon->combinedMuon(), iGmr);
		ResidualRefitting::muonInfo( storeTrk, muon->track()		, iGmr);

		SteppingHelixPropagator inwardProp  ( theField, oppositeToMomentum	);
		SteppingHelixPropagator outwardProp ( theField, alongMomentum		);
		FreeTrajectoryState recoStart = freeTrajStateMuon(muon->combinedMuon());

		
		int subDetectorLast = -1;
		for(trackingRecHit_iterator rec =  muon->track()->recHitsBegin(); 
		 rec != muon->track()->recHitsEnd(); rec++, iRec++) {
			
			if (!(*rec)->isValid() ) continue;
			DetId detid = (*rec)->geographicalId(); 

			int detector	= -1;
			int subdetector = -1;
			int blade		= -1;
			int disk		= -1;
			int ladder		= -1;
			int layer		= -1;
			int module		= -1;
			int panel		= -1;
			int ring		= -1;
			int side		= -1;
			int wheel		= -1;
			
			
			double gpX		= -1;
			double gpY		= -1;
			double gpZ		= -1;
			double gpEta	= -1;
			double gpPhi	= -1;
			
//Detector Info

			detector = detid.det();
			subdetector = detid.subdetId();
			if (subdetector != omitSystem) continue;
			
			if (subdetector != subDetectorLast) {
				subDetectorLast = subdetector;
				if (dump_) std::cout<<std::endl;
			}
			
			if (detector != DetId::Tracker) { 
				std::cout<<"OMFG NOT THE TRACKER\n"<<std::endl;
				continue;
			}

			if (dump_) std::cout<<"Tracker:: ";
			if (subdetector == ResidualRefitting::PXB) {
				PXBDetId id(detid.rawId());
				layer	= id.layer();
				ladder	= id.ladder();
				module	= id.module();
				if (dump_)	std::cout	<<	"PXB"
										<<	"\tlayer = "	<< layer
										<<	"\tladder = "	<< ladder
										<<	"\tmodule = "	<< module;
				
			} 
			else if (subdetector == ResidualRefitting::PXF) {
				PXFDetId id(detid.rawId());
				side	= id.side();
				disk	= id.disk();
				blade	= id.blade();
				panel	= id.panel();
				module	= id.module();
				if (dump_)	std::cout	<<  "PXF"
										<<	"\tside = "		<< side
										<<	"\tdisk = "		<< disk
										<<	"\tblade = "	<< blade
										<<	"\tpanel = "	<< panel
										<<	"\tmodule = "	<< module;
							
			}
			else if (subdetector == ResidualRefitting::TIB) {
				TIBDetId id(detid.rawId());
				layer	= id.layer();
				module	= id.module();
				if (dump_)	std::cout	<< "TIB"
										<< "\tlayer = "	<< layer
										<< "\tmodule = "<< module;
			}
			else if (subdetector == ResidualRefitting::TID) {
				TIDDetId id(detid.rawId());
				side	= id.side();
				wheel	= id.wheel();
				ring	= id.ring();
				if (dump_)	std::cout	<<"TID"
										<< "\tside = "	<< side
										<< "\twheel = "	<< wheel
										<< "\tring = "	<< ring;
			
			}
			else if (subdetector == ResidualRefitting::TOB) {
				TOBDetId id(detid.rawId());
				layer	= id.layer();
				module	= id.module();
				if (dump_)	std::cout	<<"TOB"
										<<"\tlayer = "	<< layer
										<<"\tmodule = "	<< module;
			
			}
			else if (subdetector == ResidualRefitting::TEC) {
				TECDetId id(detid.rawId());
				ring	= id.ring();
				module	= id.module();
				if (dump_)	std::cout	<<"TEC"
										<< "\tring = "	<< ring
										<< "\tmodule = "<< module;
			}

//Global Point Info		
			MuonTransientTrackingRecHit::MuonRecHitPointer mrhp =	
			 MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((**rec).geographicalId()), &(**rec)); 			
				GlobalPoint gp = mrhp->globalPosition();							
				gpX	= gp.x();
				gpY	= gp.y();
				gpZ	= gp.z();
				gpEta = gp.eta();
				gpPhi = gp.phi();
						
			if (dump_)	std::cout	<< setprecision(2)
//									<<	"\tX = "		<< gpX
//									<<	"\tY = "		<< gpY
									<<	"\tZ = "		<< gpZ
									<<  " \trho = "		<< sqrt( gpX * gpX + gpY * gpY )
//									<<	"\tEta = "		<< gpEta
//									<<	"\tPhi = "		<< gpPhi
									<< std::endl;
///////
//
///////
			int pars[3] = {iExtrap, iGmr, iRec};

			trkExtrap(storeExtrap, detid, pars, outwardProp, recoStart);
			iExtrap++;			
		}				
	}
	storeGmr.n_		= iGmr;
	storeTrk.n_		= iGmr;
	storeExtrap.n_	= iExtrap;
	
}							
// 
// Fill a track extrapolation 
// 
void ResidualRefitting::trkExtrap(ResidualRefitting::storage_trackExtrap& storeTemp, DetId detid, int* pars, SteppingHelixPropagator prop, FreeTrajectoryState freeTrajState) {
	bool dump_ = debug_;
	
	if (dump_) std::cout<< "In the trkExtrap function"<<std::endl;
	int		iTrk	= (pars[0]);
	int		iGmr	= (pars[1]);
	int		iRec	= (pars[2]);

	double gpExtrapX	= -99999;
	double gpExtrapY	= -99999;
	double gpExtrapZ	= -99999;
	double gpExtrapEta	= -99999;
	double gpExtrapPhi	= -99999;

	double lpX			= -99999;
	double lpY			= -99999;
	double lpZ			= -99999;

	const GeomDet* gdet = theService->trackingGeometry()->idToDet(detid);
	
	TrajectoryStateOnSurface surfTest =  prop.propagate(freeTrajState, gdet->surface());
	
	if (surfTest.isValid()) {
	
		GlobalPoint globTest	= surfTest.globalPosition();		
		gpExtrapX				= globTest.x();
		gpExtrapY				= globTest.y();
		gpExtrapZ				= globTest.z();
		gpExtrapEta				= globTest.eta();
		gpExtrapPhi				= globTest.phi();
		LocalPoint loc			= surfTest.localPosition();
		if (detid.det() == DetId::Muon || detid.det() == DetId::Tracker) {
			lpX						= loc.x();
			lpY						= loc.y();
			lpZ						= loc.z();
		}

	}
	storeTemp.muonLink_	[iTrk] = iGmr		;
	storeTemp.recLink_	[iTrk] = iRec		;
	storeTemp.gpX_		[iTrk] = gpExtrapX	;
	storeTemp.gpY_		[iTrk] = gpExtrapY	;
	storeTemp.gpZ_		[iTrk] = gpExtrapZ	;
	storeTemp.gpEta_	[iTrk] = gpExtrapEta;
	storeTemp.gpPhi_	[iTrk] = gpExtrapPhi;	
	storeTemp.lpX_		[iTrk] = lpX		;
	storeTemp.lpY_		[iTrk] = lpY		;
	storeTemp.lpZ_		[iTrk] = lpZ		;
}
// 
// Store the SAM and Track position info at a particular rho
// 
void ResidualRefitting::cylExtrapTrkSam(int recNum, reco::TrackRef track, ResidualRefitting::storage_trackExtrap& storage, double rho) {

	Cylinder::PositionType pos(0,0,0);
	Cylinder::RotationType rot;

	Cylinder::CylinderPointer myCylinder = Cylinder::build(pos, rot, rho);
	SteppingHelixPropagator inwardProp  ( theField, oppositeToMomentum );
	SteppingHelixPropagator outwardProp ( theField, alongMomentum );
	FreeTrajectoryState recoStart = freeTrajStateMuon(track);
	TrajectoryStateOnSurface recoProp = outwardProp.propagate(recoStart, *myCylinder);

	double xVal		= -9999;
	double yVal		= -9999;
	double zVal		= -9999;
	double phiVal	= -9999;
	double etaVal	= -9999;

	if(recoProp.isValid()) {
		GlobalPoint recoPoint = recoProp.globalPosition();
		xVal = recoPoint.x();
		yVal = recoPoint.y();
		zVal = recoPoint.z();
		phiVal = recoPoint.phi();
		etaVal = recoPoint.eta();		
	}
	storage.muonLink_[recNum]	= recNum;
	storage.gpX_	[recNum]	= xVal;
	storage.gpY_	[recNum]	= yVal;
	storage.gpZ_	[recNum]	= zVal;
	storage.gpEta_	[recNum]	= etaVal;
	storage.gpPhi_	[recNum]	= phiVal;
}
//  
// zero storage
// 
void ResidualRefitting::zero_storage() {
	if (debug_)	printf("zero_storage\n");

	zero_muon(&storageGmrNew_	);
	zero_muon(&storageSamNew_	);
	zero_muon(&storageTrkNew_	);
	zero_muon(&storageGmrNoSt1_	);
	zero_muon(&storageSamNoSt1_	);
	zero_muon(&storageGmrNoSt2_	);
	zero_muon(&storageSamNoSt2_	);
	zero_muon(&storageGmrNoSt3_	);
	zero_muon(&storageSamNoSt3_	);
	zero_muon(&storageGmrNoSt4_	);
	zero_muon(&storageSamNoSt4_	);
//zero out the tracker
	zero_muon(&storageGmrNoPXBLayer1	);
	zero_muon(&storageGmrNoPXBLayer2	);
	zero_muon(&storageGmrNoPXBLayer3	);

	zero_muon(&storageGmrNoPXF	);

	zero_muon(&storageGmrNoTIBLayer1	);
	zero_muon(&storageGmrNoTIBLayer2	);
	zero_muon(&storageGmrNoTIBLayer3	);
	zero_muon(&storageGmrNoTIBLayer4	);

	zero_muon(&storageGmrNoTID	);

	zero_muon(&storageGmrNoTOBLayer1	);
	zero_muon(&storageGmrNoTOBLayer2	);
	zero_muon(&storageGmrNoTOBLayer3	);
	zero_muon(&storageGmrNoTOBLayer4	);
	zero_muon(&storageGmrNoTOBLayer5	);
	zero_muon(&storageGmrNoTOBLayer6	);

	zero_muon(&storageGmrNoTEC	);

	zero_muon(&storageTrkNoPXBLayer1	);
	zero_muon(&storageTrkNoPXBLayer2	);
	zero_muon(&storageTrkNoPXBLayer3	);

	zero_muon(&storageTrkNoPXF	);

	zero_muon(&storageTrkNoTIBLayer1	);
	zero_muon(&storageTrkNoTIBLayer2	);
	zero_muon(&storageTrkNoTIBLayer3	);
	zero_muon(&storageTrkNoTIBLayer4	);

	zero_muon(&storageTrkNoTID	);

	zero_muon(&storageTrkNoTOBLayer1	);
	zero_muon(&storageTrkNoTOBLayer2	);
	zero_muon(&storageTrkNoTOBLayer3	);
	zero_muon(&storageTrkNoTOBLayer4	);
	zero_muon(&storageTrkNoTOBLayer5	);
	zero_muon(&storageTrkNoTOBLayer6	);

	zero_muon(&storageTrkNoTEC	);

	zero_trackExtrap(&storageTrackExtrapRec_		);
	zero_trackExtrap(&storageTrackExtrapTracker_	);
	zero_trackExtrap(&storageTrackExtrapRecNoSt1_	);
	zero_trackExtrap(&storageTrackExtrapRecNoSt2_	);
	zero_trackExtrap(&storageTrackExtrapRecNoSt3_	);
	zero_trackExtrap(&storageTrackExtrapRecNoSt4_	);

	zero_trackExtrap(&trackExtrap120_				);

	zero_trackExtrap(&samExtrap120_					);

	zero_trackExtrap(&storageTrackNoPXBLayer1		);
	zero_trackExtrap(&storageTrackNoPXBLayer2		);
	zero_trackExtrap(&storageTrackNoPXBLayer3		);

	zero_trackExtrap(&storageTrackNoPXF				);

	zero_trackExtrap(&storageTrackNoTIBLayer1		);
	zero_trackExtrap(&storageTrackNoTIBLayer2		);
	zero_trackExtrap(&storageTrackNoTIBLayer3		);
	zero_trackExtrap(&storageTrackNoTIBLayer4		);

	zero_trackExtrap(&storageTrackNoTOBLayer1		);
	zero_trackExtrap(&storageTrackNoTOBLayer2		);
	zero_trackExtrap(&storageTrackNoTOBLayer3		);
	zero_trackExtrap(&storageTrackNoTOBLayer4		);
	zero_trackExtrap(&storageTrackNoTOBLayer5		);
	zero_trackExtrap(&storageTrackNoTOBLayer6		);

	zero_trackExtrap(&storageTrackNoTEC				);

	zero_trackExtrap(&storageTrackNoTID				);

	storageRecMuon_		.n_ = 0;
	storageTrackHit_	.n_ = 0;

	for ( int i = 0; i < ResidualRefitting::N_MAX_STORED_HIT; i++) {

//Muon Rec Hits		
		storageRecMuon_.muonLink_		[i]= -99999;
		
		storageRecMuon_.system_			[i]= -99999;
		storageRecMuon_.endcap_			[i]= -99999;
		storageRecMuon_.station_		[i]= -99999;
		storageRecMuon_.ring_			[i]= -99999;
		storageRecMuon_.chamber_		[i]= -99999;
		storageRecMuon_.layer_			[i]= -99999;
		storageRecMuon_.superLayer_		[i]= -99999;
		
		storageRecMuon_.	gpX_		[i]= -99999;
		storageRecMuon_.	gpY_		[i]= -99999;
		storageRecMuon_.	gpZ_		[i]= -99999;
		storageRecMuon_.	gpEta_		[i]= -99999;
		storageRecMuon_.	gpPhi_		[i]= -99999;
		storageRecMuon_.	lpX_		[i]= -99999;
		storageRecMuon_.	lpY_		[i]= -99999;
//Tracker Rec Hits
		
		storageTrackHit_.muonLink_		[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.detector_		[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.subdetector_	[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.blade_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.disk_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.ladder_		[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.layer_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.module_		[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.panel_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.ring_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.side_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_.wheel_			[N_MAX_STORED_HIT] = -99999;
				
		storageTrackHit_. gpX_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_. gpY_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_. gpZ_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_. gpEta_		[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_. gpPhi_		[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_. lpX_			[N_MAX_STORED_HIT] = -99999;
		storageTrackHit_. lpY_			[N_MAX_STORED_HIT] = -99999;
	}
}
//
// Zero out a muon reference
//
void ResidualRefitting::zero_muon(ResidualRefitting::storage_muon* str){

	str->n_ = 0;
	
	for (int i = 0; i < ResidualRefitting::N_MAX_STORED; i++) {
    
		str->pt_  [i] = -9999;
		str->eta_ [i] = -9999;
		str->p_   [i] = -9999;
		str->phi_ [i] = -9999;	
	}

}
//
// Zero track extrapolation
//
void ResidualRefitting::zero_trackExtrap(ResidualRefitting::storage_trackExtrap* str) {
	
	str->n_ = 0;
	for (int i = 0; i < ResidualRefitting::N_MAX_STORED_HIT; i++) {

		str->muonLink_	[i] = -9999;
		str->recLink_	[i] = -9999;
		str->gpX_		[i] = -9999;
		str->gpY_		[i] = -9999;
		str->gpZ_		[i] = -9999;
		str->gpEta_		[i] = -9999;
		str->gpPhi_		[i] = -9999;
		str->lpX_		[i] = -9999;
		str->lpY_		[i] = -9999;
		str->lpZ_		[i] = -9999;
	}
	
}
//  
// Begin Job
// 
void ResidualRefitting::beginJob( const EventSetup& ) {

	std::cout<<"Creating file "<< outputFileName_.c_str()<<std::endl;

	outputFile_  = new TFile( outputFileName_.c_str(), "RECREATE" ) ; 

	outputTree_ = new TTree("outputTree","outputTree");

	ResidualRefitting::branchMuon(storageGmrNew_	, "gmrNew");
	ResidualRefitting::branchMuon(storageGmrNoSt1_	, "gmrNoSt1");
	ResidualRefitting::branchMuon(storageGmrNoSt2_	, "gmrNoSt2");
	ResidualRefitting::branchMuon(storageGmrNoSt3_	, "gmrNoSt3");
	ResidualRefitting::branchMuon(storageGmrNoSt4_	, "gmrNoSt4");

	ResidualRefitting::branchMuon(storageSamNew_	, "samNew");
	ResidualRefitting::branchMuon(storageSamNoSt1_	, "samNoSt1");
	ResidualRefitting::branchMuon(storageSamNoSt2_	, "samNoSt2");
	ResidualRefitting::branchMuon(storageSamNoSt3_	, "samNoSt3");
	ResidualRefitting::branchMuon(storageSamNoSt4_	, "samNoSt4");

	ResidualRefitting::branchMuon(storageTrkNew_	, "trkNew");
	ResidualRefitting::branchMuon(storageGmrNoPXBLayer1	, "gmrNoPXBLayer1");
	ResidualRefitting::branchMuon(storageGmrNoPXBLayer2	, "gmrNoPXBLayer2");
	ResidualRefitting::branchMuon(storageGmrNoPXBLayer3	, "gmrNoPXBLayer3");
	ResidualRefitting::branchMuon(storageGmrNoPXF	, "gmrNoPXF");
	ResidualRefitting::branchMuon(storageGmrNoTIBLayer1	, "gmrNoTIBLayer1");
	ResidualRefitting::branchMuon(storageGmrNoTIBLayer2	, "gmrNoTIBLayer2");
	ResidualRefitting::branchMuon(storageGmrNoTIBLayer3	, "gmrNoTIBLayer3");
	ResidualRefitting::branchMuon(storageGmrNoTIBLayer4	, "gmrNoTIBLayer4");
	ResidualRefitting::branchMuon(storageGmrNoTID	, "gmrNoTID");
	ResidualRefitting::branchMuon(storageGmrNoTOBLayer1	, "gmrNoTOBLayer1");
	ResidualRefitting::branchMuon(storageGmrNoTOBLayer2	, "gmrNoTOBLayer2");
	ResidualRefitting::branchMuon(storageGmrNoTOBLayer3	, "gmrNoTOBLayer3");
	ResidualRefitting::branchMuon(storageGmrNoTOBLayer4	, "gmrNoTOBLayer4");
	ResidualRefitting::branchMuon(storageGmrNoTOBLayer5	, "gmrNoTOBLayer5");
	ResidualRefitting::branchMuon(storageGmrNoTOBLayer6	, "gmrNoTOBLayer6");
	ResidualRefitting::branchMuon(storageGmrNoTEC	, "gmrNoTEC");

	ResidualRefitting::branchMuon(storageTrkNoPXBLayer1	, "trkNoPXBLayer1");
	ResidualRefitting::branchMuon(storageTrkNoPXBLayer2	, "trkNoPXBLayer2");
	ResidualRefitting::branchMuon(storageTrkNoPXBLayer3	, "trkNoPXBLayer3");
	ResidualRefitting::branchMuon(storageTrkNoPXF	, "trkNoPXF");
	ResidualRefitting::branchMuon(storageTrkNoTIBLayer1	, "trkNoTIBLayer1");
	ResidualRefitting::branchMuon(storageTrkNoTIBLayer2	, "trkNoTIBLayer2");
	ResidualRefitting::branchMuon(storageTrkNoTIBLayer3	, "trkNoTIBLayer3");
	ResidualRefitting::branchMuon(storageTrkNoTIBLayer4	, "trkNoTIBLayer4");
	ResidualRefitting::branchMuon(storageTrkNoTID	, "trkNoTID");
	ResidualRefitting::branchMuon(storageTrkNoTOBLayer1	, "trkNoTOBLayer1");
	ResidualRefitting::branchMuon(storageTrkNoTOBLayer2	, "trkNoTOBLayer2");
	ResidualRefitting::branchMuon(storageTrkNoTOBLayer3	, "trkNoTOBLayer3");
	ResidualRefitting::branchMuon(storageTrkNoTOBLayer4	, "trkNoTOBLayer4");
	ResidualRefitting::branchMuon(storageTrkNoTOBLayer5	, "trkNoTOBLayer5");
	ResidualRefitting::branchMuon(storageTrkNoTOBLayer6	, "trkNoTOBLayer6");
	ResidualRefitting::branchMuon(storageTrkNoTEC	, "trkNoTEC");
					
	outputBranch_ = outputTree_ -> Branch("recHitsNew", &storageRecMuon_, 

		"n_/I:"
		"muonLink_[100]/I:"
		
		"system_[100]/I:"
		"endcap_[100]/I:"
		"station_[100]/I:"
		"ring_[100]/I:"
		"chamber_[100]/I:"
		"layer_[100]/I:"
		"superLayer_[100]/I:"
		
	
		"gpX_[100]/F:"
		"gpY_[100]/F:"
		"gpZ_[100]/F:"
		"gpEta_[100]/F:"
		"gpPhi_[100]/F:"
		"lpX_[100]/F:"
		"lpY_[100]/F:"
		"lpZ_[100]/F"
		);
		
		
	outputBranch_ = outputTree_ -> Branch("recHitsTracker", &storageTrackHit_,
	
		"n_/I:"
		
		"muonLink_[100]/I:"
		"detector_[100]/I:"
		"subdetector_[100]/I:"
		"blade_[100]/I:"
		"disk_[100]/I:"
		"ladder_[100]/I:"
		"layer_[100]/I:"
		"module_[100]/I:"
		"panel_[100]/I:"
		"ring_[100]/I:"
		"side_[100]/I:"
		"wheel_[100]/I:"
				
		"gpX_[100]/F:"
		"gpY_[100]/F:"
		"gpZ_[100]/F:"
		"gpEta_[100]/F:"
		"gpPhi_[100]/F:"
		"lpX_[100]/F:"
		"lpY_[100]/F:"
		"lpZ_[100]/F"
		);	
		
		
	ResidualRefitting::branchTrackExtrap(storageTrackExtrapRec_		, "trkExtrap");
	ResidualRefitting::branchTrackExtrap(storageTrackExtrapRecNoSt1_	, "trkExtrapNoSt1");
	ResidualRefitting::branchTrackExtrap(storageTrackExtrapRecNoSt2_	, "trkExtrapNoSt2");
	ResidualRefitting::branchTrackExtrap(storageTrackExtrapRecNoSt3_	, "trkExtrapNoSt3");
	ResidualRefitting::branchTrackExtrap(storageTrackExtrapRecNoSt4_	, "trkExtrapNoSt4");

	ResidualRefitting::branchTrackExtrap(storageTrackExtrapTracker_, "trkExtrapTracker");
	ResidualRefitting::branchTrackExtrap(storageTrackNoPXF			, "trkExtrapNoPXF");
	ResidualRefitting::branchTrackExtrap(storageTrackNoPXBLayer1			, "trkExtrapNoPXBLayer1");
	ResidualRefitting::branchTrackExtrap(storageTrackNoPXBLayer2			, "trkExtrapNoPXBLayer2");
	ResidualRefitting::branchTrackExtrap(storageTrackNoPXBLayer3			, "trkExtrapNoPXBLayer3");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTIBLayer1			, "trkExtrapNoTIBLayer1");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTIBLayer2			, "trkExtrapNoTIBLayer2");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTIBLayer3			, "trkExtrapNoTIBLayer3");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTIBLayer4			, "trkExtrapNoTIBLayer4");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTID			, "trkExtrapNoTID");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer1			, "trkExtrapNoTOBLayer1");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer2			, "trkExtrapNoTOBLayer2");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer3			, "trkExtrapNoTOBLayer3");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer4			, "trkExtrapNoTOBLayer4");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer5			, "trkExtrapNoTOBLayer5");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer6			, "trkExtrapNoTOBLayer6");
	ResidualRefitting::branchTrackExtrap(storageTrackNoTEC			, "trkExtrapNoTEC");

	ResidualRefitting::branchTrackExtrap(trackExtrap120_			, "trackCyl120");
	ResidualRefitting::branchTrackExtrap(samExtrap120_				, "samCyl120");

}
//
// Set the Muon Branches
//
void ResidualRefitting::branchMuon(ResidualRefitting::storage_muon& storageTmp, std::string branchName){

	outputBranch_ = outputTree_ -> Branch(branchName.c_str(), &storageTmp, 
									"n_/I:"
									"charge_[5]/I:"
									"pt_[5]/F:"
									"eta_[5]/F:"
									"p_[5]/F:"
									"phi_[5]/F"
										
										);

}
//
// Set the Muon Branches
//
void ResidualRefitting::branchTrackExtrap(ResidualRefitting::storage_trackExtrap& storageTmp, std::string branchName){

	outputBranch_ = outputTree_ -> Branch(branchName.c_str(), &storageTmp, 
									"n_/I:"
									"muonLink_[100]/I:"
									"recLink_[100]/I:"
									"gpX_[100]/F:"
									"gpY_[100]/F:"
									"gpZ_[100]/F:"
									"gpEta_[100]/F:"
									"gpPhi_[100]/F:"
									"lpX_[100]/F:"
									"lpY_[100]/F:"
									"lpZ_[100]/F"
										
									);

}
// 
// End Job
// 
void ResidualRefitting::endJob() {


  outputFile_ -> Write();

  outputFile_ -> Close() ;

}
// 
// Return a Free Trajectory state for a muon track
// 
FreeTrajectoryState ResidualRefitting::freeTrajStateMuon(reco::TrackRef muon){
					
			math::XYZPoint  innerPos = muon -> referencePoint();
			math::XYZVector innerMom = muon -> momentum();
			if (debug_) std::cout	<<  "Inner Pos: "
									<<	"\tx = "	<< innerPos.X()
									<<	"\ty = "	<< innerPos.Y()
									<<	"\tz = "	<< innerPos.Z()
									<<	std::endl;
								
			GlobalPoint   innerPoint( innerPos.X(), innerPos.Y(), innerPos.Z());
			GlobalVector  innerVec  ( innerMom.X(), innerMom.Y(), innerMom.Z());

			FreeTrajectoryState recoStart( innerPoint, innerVec, muon ->charge(), theField ); 
			return recoStart;

}

// 
// nTuple value Dumps
// 
// dump Track Extrapolation
void ResidualRefitting::dumpTrackExtrap(ResidualRefitting::storage_trackExtrap track) {
	std::cout<<"\n\nExtrapolation Dump:\n";
	for (unsigned int i = 0; i < (unsigned int)track.n_; i++) {
		
//		double rho = sqrt( (float)track.gpX_[i] * (float)track.gpX_[i] + (float)track.gpY_[i] * (float)track.gpY_[i]  );
		
		printf ("%d\trecLink = %d", i,	(int)track.recLink_[i]	);
//		printf ("\tGlobal\tx = %0.3f"		,		(float)track.gpX_[i]	);
//		printf ("\ty = %0.3f"		,		(float)track.gpY_[i]	);
//		printf ("\tz = %0.3f"		,		(float)track.gpZ_[i]	);
//		printf ("\trho =%0.3f"		,		rho						);
//		printf ("\teta = %0.3f"		,		(float)track.gpEta_[i]	);
//		printf ("\tphi = %0.3f"		,		(float)track.gpPhi_[i]	);
		printf ("\t\tLocal\tx = %0.3f"	,		(float)track.lpX_[i]	);
		printf ("\ty = %0.3f"		,	(float)track.lpY_[i]		);
		printf ("\tz = %0.3f\n"	,	(float)track.lpZ_[i]		);
		
	}
	
}
// dump Muon Rec Hits
void ResidualRefitting::dumpMuonRecHits(ResidualRefitting::storage_hit hit) {
	std::cout<<"Muon Rec Hits Dump:\n";
	for (unsigned int i = 0; i < (unsigned int)hit.n_; i++) {
		
//		double rho = sqrt( (float)hit.gpX_[i] * (float)hit.gpX_[i] + (float)hit.gpY_[i] * (float)hit.gpY_[i]  );
		
		printf ("%d\tsubdetector = %d\t superLayer =%d"	, i,	(int)hit.system_[i], (int)hit.superLayer_[i]	);
//		printf ("\tGlobal\tx = %0.3f"			,		(float)hit.gpX_[i]			);
//		printf ("\ty = %0.3f"				,		(float)hit.gpY_[i]			);
//		printf ("\tz = %0.3f"				,		(float)hit.gpZ_[i]			);
//		printf ("\trho =%0.3f"				,		rho							);
//		printf ("\teta = %0.3f"				,		(float)hit.gpEta_[i]		);
//		printf ("\tphi = %0.3f\n"			,		(float)hit.gpPhi_[i]		);
		printf ("\t\tLocal\tx = %0.3f"		,		(float)hit.lpX_[i]			);
		printf ("\ty = %0.3f"				,		(float)hit.lpY_[i]			);
		printf ("\tz = %0.3f\n"			,		(float)hit.lpZ_[i]			);
		
	}
	
}
// dump Tracker Rec Hits
void ResidualRefitting::dumpTrackHits(ResidualRefitting::storage_trackHit hit) {
	std::cout<<"Tracker Rec Hits Dump:\n";
	for (unsigned int i = 0; i < (unsigned int)hit.n_; i++) {
		
//		double rho = sqrt( (float)hit.gpX_[i] * (float)hit.gpX_[i] + (float)hit.gpY_[i] * (float)hit.gpY_[i]  );
		
		printf ("%d\tsubdetector = %d"		, i,	(int)hit.subdetector_[i]	);
		printf ("\tlayer = %d"				, 	(int)hit.layer_[i]	);
//		printf ("\tGlobal\tx = %0.3f"			,		(float)hit.gpX_[i]			);
//		printf ("\ty = %0.3f"				,		(float)hit.gpY_[i]			);
//		printf ("\tz = %0.3f"				,		(float)hit.gpZ_[i]			);
//		printf ("\trho =%0.3f"				,		rho							);
//		printf ("\teta = %0.3f"				,		(float)hit.gpEta_[i]		);
//		printf ("\tphi = %0.3f\n"			,		(float)hit.gpPhi_[i]		);
		printf ("\t\tLocal\tx = %0.3f"		,		(float)hit.lpX_[i]			);
		printf ("\ty = %0.3f"				,		(float)hit.lpY_[i]			);
		printf ("\tz = %0.3f\n"				,		(float)hit.lpZ_[i]			);
		
	}
	
}
// Dump p, pt, eta, phi for a muon
void ResidualRefitting::dumpRecoMuonColl(reco::MuonCollection::const_iterator muon) {

	float pt = muon->pt();
	float p  = muon->p  ();
	float eta = muon->eta();
	float phi = muon->phi();
	printf("\tgmr: \tp = %0.0f \t pt = %0.0f \t eta = %0.2f \t phi = %0.2f\n", p, pt, eta, phi);

	pt = muon->standAloneMuon()->pt();
	p  = muon->standAloneMuon()->p  ();
	eta = muon->standAloneMuon()->eta();
	phi = muon->standAloneMuon()->phi();

	printf("\tsam: \tp = %0.0f \t pt = %0.0f \t eta = %0.2f \t phi = %0.2f\n", p, pt, eta, phi);
	
	pt = muon->track()->pt();
	p  = muon->track()->p();
	eta = muon->track()->eta();
	phi = muon->track()->phi();

			
	printf("\ttrk: \tp = %0.0f \t pt = %0.0f \t eta = %0.2f \t phi = %0.2f\n", p, pt, eta, phi);

	pt = muon->combinedMuon()->pt();
	p  = muon->combinedMuon()->p  ();
	eta = muon->combinedMuon()->eta();
	phi = muon->combinedMuon()->phi();

	printf("\tcmb: \tp = %0.0f \t pt = %0.0f \t eta = %0.2f \t phi = %0.2f\n", p, pt, eta, phi);


}


DEFINE_FWK_MODULE( ResidualRefitting );
