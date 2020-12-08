#include <iostream>
#include "ResidualRefitting.h"
#include <iomanip>

//framework includes
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

ResidualRefitting::ResidualRefitting(const edm::ParameterSet& cfg)
    : magFieldToken_(esConsumes()),
      topoToken_(esConsumes()),
      trackingGeometryToken_(esConsumes()),
      propagatorToken_(esConsumes(edm::ESInputTag("", cfg.getParameter<std::string>("propagator")))),
      outputFileName_(cfg.getUntrackedParameter<std::string>("histoutputFile")),
      muons_(cfg.getParameter<edm::InputTag>("muons")),
      muonsRemake_(cfg.getParameter<edm::InputTag>("muonsRemake")),  //This Feels Misalignment
      muonsNoStation1_(cfg.getParameter<edm::InputTag>("muonsNoStation1")),
      muonsNoStation2_(cfg.getParameter<edm::InputTag>("muonsNoStation2")),
      muonsNoStation3_(cfg.getParameter<edm::InputTag>("muonsNoStation3")),
      muonsNoStation4_(cfg.getParameter<edm::InputTag>("muonsNoStation4")),

      /*
  muonsNoPXBLayer1_		( cfg.getParameter<edm::InputTag>("muonsNoPXBLayer1"	) ),
  muonsNoPXBLayer2_		( cfg.getParameter<edm::InputTag>("muonsNoPXBLayer1"	) ),
  muonsNoPXBLayer3_		( cfg.getParameter<edm::InputTag>("muonsNoPXBLayer1"	) ),

  muonsNoTIBLayer1_			( cfg.getParameter<edm::InputTag>("muonsNoTIBLayer1"	) ),
  muonsNoTIBLayer2_			( cfg.getParameter<edm::InputTag>("muonsNoTIBLayer2"	) ),
  muonsNoTIBLayer3_			( cfg.getParameter<edm::InputTag>("muonsNoTIBLayer3"	) ),
  muonsNoTIBLayer4_			( cfg.getParameter<edm::InputTag>("muonsNoTIBLayer4"	) ),

  muonsNoTOBLayer1_			( cfg.getParameter<edm::InputTag>("muonsNoTOBLayer1"	) ),
  muonsNoTOBLayer2_			( cfg.getParameter<edm::InputTag>("muonsNoTOBLayer2"	) ),
  muonsNoTOBLayer3_			( cfg.getParameter<edm::InputTag>("muonsNoTOBLayer3"	) ),
  muonsNoTOBLayer4_			( cfg.getParameter<edm::InputTag>("muonsNoTOBLayer4"	) ),
  muonsNoTOBLayer5_			( cfg.getParameter<edm::InputTag>("muonsNoTOBLayer5"	) ),
  muonsNoTOBLayer6_			( cfg.getParameter<edm::InputTag>("muonsNoTOBLayer6"	) ),*/
      debug_(cfg.getUntrackedParameter<bool>("doDebug")),
      outputFile_(nullptr),
      outputTree_(nullptr),
      outputBranch_(nullptr),
      theField(nullptr) {
  eventInfo_.evtNum_ = 0;
  eventInfo_.evtNum_ = 0;

  // service parameters
  edm::ParameterSet serviceParameters = cfg.getParameter<edm::ParameterSet>("ServiceParameters");

  // the services
  theService = new MuonServiceProxy(serviceParameters, consumesCollector());

}  //The constructor

void ResidualRefitting::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  if (debug_)
    printf("STARTING EVENT\n");

  eventInfo_.evtNum_ = (int)event.id().run();
  eventInfo_.runNum_ = (int)event.id().event();

  // Generator Collection

  // The original muon collection that is sitting in memory
  edm::Handle<reco::MuonCollection> muons;

  edm::Handle<reco::TrackCollection> muonTracks;
  edm::Handle<reco::TrackCollection> muonsNoSt1;
  edm::Handle<reco::TrackCollection> muonsNoSt2;
  edm::Handle<reco::TrackCollection> muonsNoSt3;
  edm::Handle<reco::TrackCollection> muonsNoSt4;

  event.getByLabel(muons_, muons);  //set label to muons
  event.getByLabel(muonsRemake_, muonTracks);
  event.getByLabel(muonsNoStation1_, muonsNoSt1);
  event.getByLabel(muonsNoStation2_, muonsNoSt2);
  event.getByLabel(muonsNoStation3_, muonsNoSt3);
  event.getByLabel(muonsNoStation4_, muonsNoSt4);

  /*
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
*/
  //magnetic field information
  theField = &eventSetup.getData(magFieldToken_);
  edm::ESHandle<GlobalTrackingGeometry> globalTrackingGeometry = eventSetup.getHandle(trackingGeometryToken_);
  thePropagator = eventSetup.getHandle(propagatorToken_);
  theService->update(eventSetup);

  //Zero storage
  zero_storage();

  //Do the Gmr Muons from the unModified Collection

  /*
	int iGmr = 0;
	if ( (muons->end() - muons->begin()) > 0) printf("Data Dump:: Original GMR Muons\n");
	for ( MuonCollection::const_iterator muon = muons->begin(); muon!=muons->end(); muon++, iGmr++) {
		if ( iGmr >= ResidualRefitting::N_MAX_STORED) break; // error checking
		if (!debug
		
		dumpTrackRef(muon->combinedMuon(), "cmb"); 
		dumpTrackRef(muon->standAloneMuon(), "sam");
		dumpTrackRef(muon->track(), "trk");
		

	}
	storageGmrOld_.n_ = iGmr;
	storageSamNew_.n_ = iGmr;
*/

  //Refitted muons
  if (debug_)
    printf("Data Dump:: Rebuilt GMR Muon Track With TeV refitter default\n");
  int iGmrRemake = 0;
  for (reco::TrackCollection::const_iterator muon = muonTracks->begin(); muon != muonTracks->end();
       muon++, iGmrRemake++) {
    if (iGmrRemake >= ResidualRefitting::N_MAX_STORED)
      break;  // error checking
              // from TrackInfoProducer/test/TrackInfoAnalyzerExample.cc
    reco::TrackRef trackref = reco::TrackRef(muonTracks, iGmrRemake);

    if (debug_)
      dumpTrackRef(trackref, "gmr");
    muonInfo(storageGmrNew_, trackref, iGmrRemake);
  }
  storageGmrNew_.n_ = iGmrRemake;

  if (debug_)
    printf("muons Remake");
  if (debug_)
    printf("-----------------------------------------\n");
  CollectTrackHits(muonTracks, storageTrackExtrapRec_, eventSetup);

  if (true) {
    printf("muons No Station 1");
    printf("-----------------------------------------\n");
  }
  NewTrackMeasurements(muonTracks, muonsNoSt1, storageTrackExtrapRecNoSt1_);

  if (true) {
    printf("muons No Station 2");
    printf("-----------------------------------------\n");
  }
  NewTrackMeasurements(muonTracks, muonsNoSt2, storageTrackExtrapRecNoSt2_);

  if (true) {
    printf("muons No Station 3");
    printf("-----------------------------------------\n");
  }
  NewTrackMeasurements(muonTracks, muonsNoSt3, storageTrackExtrapRecNoSt3_);

  if (true) {
    printf("muons No Station 4");
    printf("-----------------------------------------\n");
  }
  NewTrackMeasurements(muonTracks, muonsNoSt4, storageTrackExtrapRecNoSt4_);

  //	dumpMuonRecHits(storageRecMuon_);

  /****************************************************************************************************************************************/

  /*
 *	extrapolates track to a cylinder.
 *  commented for cosmic runs with no tracker in reco muons!!
 *
*/

  int iGmrCyl = 0;
  for (reco::MuonCollection::const_iterator muon = muons->begin(); muon != muons->end(); muon++, iGmrCyl++) {
    dumpTrackRef(muon->combinedMuon(), "cmb");
    dumpTrackRef(muon->standAloneMuon(), "sam");
    dumpTrackRef(muon->track(), "trk");

    cylExtrapTrkSam(iGmrCyl, muon->standAloneMuon(), samExtrap120_, 120.);
    cylExtrapTrkSam(iGmrCyl, muon->track(), trackExtrap120_, 120.);
  }
  samExtrap120_.n_ = iGmrCyl;
  trackExtrap120_.n_ = iGmrCyl;

  if (iGmrRemake > 0 || iGmrCyl > 0) {
    outputTree_->Fill();
    std::cout << "FILLING NTUPLE!" << std::endl;
    std::cout << "Entries Recorded: " << outputTree_->GetEntries() << " Branch :: " << outputBranch_->GetEntries()
              << std::endl
              << std::endl;
  } else
    std::cout << "no tracks -- no fill!\n" << std::endl << std::endl;

  //  /*************************************************************************************************************/
  //  //END OF ntuple dumper
  //  //END OF ntuple dumper
  //  /***********************************************************************************************************/
}
//end Analyze() main function

//------------------------------------------------------------------------------
//
// Destructor
//
ResidualRefitting::~ResidualRefitting() {
  delete outputFile_;
  delete theService;
}
//
// Track Collection Analysis
//
void ResidualRefitting::CollectTrackHits(edm::Handle<reco::TrackCollection> trackColl,
                                         ResidualRefitting::storage_trackExtrap& trackExtrap,
                                         const edm::EventSetup& eventSetup) {
  //Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &eventSetup.getData(topoToken_);

  int iMuonHit = 0;
  int iTrackHit = 0;
  int numTracks = 0;

  for (reco::TrackCollection::const_iterator muon = trackColl->begin(); muon != trackColl->end(); muon++) {
    int iTrack = muon - trackColl->begin();
    reco::TrackRef trackref = reco::TrackRef(trackColl, iTrack);
    FreeTrajectoryState recoStart = ResidualRefitting::freeTrajStateMuon(trackref);

    if (debug_)
      dumpTrackRef(trackref, "CollectTrackHits Track");

    int iRec = 0;
    for (auto const& rec : muon->recHits()) {
      DetId detid = rec->geographicalId();

      if (detid.det() != DetId::Muon && detid.det() != DetId::Tracker) {
        if (debug_)
          printf("Rec Hit not from muon system or tracker... continuing...\n");
        continue;
      }
      //			numTracks++;
      // Get Local and Global Position of Hits

      LocalPoint lp = rec->localPosition();
      float lpX = lp.x();
      float lpY = lp.y();
      float lpZ = lp.z();

      auto mrhp = MuonTransientTrackingRecHit::specificBuild(
          theService->trackingGeometry()->idToDet(rec->geographicalId()), rec);

      GlobalPoint gp = mrhp->globalPosition();
      float gpRecX = gp.x();
      float gpRecY = gp.y();
      float gpRecZ = gp.z();
      float gpRecEta = gp.eta();
      float gpRecPhi = gp.phi();

      if (detid.det() == DetId::Muon) {
        int systemMuon = detid.subdetId();  // 1 DT; 2 CSC; 3 RPC
        int endcap = -999;
        int station = -999;
        int ring = -999;
        int chamber = -999;
        int layer = -999;
        int superLayer = -999;
        int wheel = -999;
        int sector = -999;
        if (systemMuon == MuonSubdetId::CSC) {
          CSCDetId id(detid.rawId());
          endcap = id.endcap();
          station = id.station();
          ring = id.ring();
          chamber = id.chamber();
          layer = id.layer();
          if (debug_)
            printf("CSC\t[endcap][station][ringN][chamber][layer]:[%d][%d][%d][%d][%d]\t",
                   endcap,
                   station,
                   ring,
                   chamber,
                   layer);

        } else if (systemMuon == MuonSubdetId::DT) {
          DTWireId id(detid.rawId());
          station = id.station();
          layer = id.layer();
          superLayer = id.superLayer();
          wheel = id.wheel();
          sector = id.sector();
          if (debug_)
            printf("DT \t[station][layer][superlayer]:[%d][%d][%d]\n", station, layer, superLayer);

        } else if (systemMuon == MuonSubdetId::RPC) {
          RPCDetId id(detid.rawId());
          station = id.station();
          if (debug_)
            printf("RPC\t[station]:[%d]\n", station);
        }

        storageRecMuon_.muonLink_[iMuonHit] = iTrack;
        storageRecMuon_.system_[iMuonHit] = systemMuon;
        storageRecMuon_.endcap_[iMuonHit] = endcap;
        storageRecMuon_.station_[iMuonHit] = station;
        storageRecMuon_.ring_[iMuonHit] = ring;
        storageRecMuon_.chamber_[iMuonHit] = chamber;
        storageRecMuon_.layer_[iMuonHit] = layer;
        storageRecMuon_.superLayer_[iMuonHit] = superLayer;
        storageRecMuon_.wheel_[iMuonHit] = wheel;
        storageRecMuon_.sector_[iMuonHit] = sector;

        storageRecMuon_.gpX_[iMuonHit] = gpRecX;
        storageRecMuon_.gpY_[iMuonHit] = gpRecY;
        storageRecMuon_.gpZ_[iMuonHit] = gpRecZ;
        storageRecMuon_.gpEta_[iMuonHit] = gpRecEta;
        storageRecMuon_.gpPhi_[iMuonHit] = gpRecPhi;
        storageRecMuon_.lpX_[iMuonHit] = lpX;
        storageRecMuon_.lpY_[iMuonHit] = lpY;
        storageRecMuon_.lpZ_[iMuonHit] = lpZ;
        iMuonHit++;

      } else if (detid.det() == DetId::Tracker) {
        if (debug_)
          printf("Tracker\n");

        StoreTrackerRecHits(detid, tTopo, iTrack, iTrackHit);

        storageTrackHit_.gpX_[iTrackHit] = gpRecX;
        storageTrackHit_.gpY_[iTrackHit] = gpRecY;
        storageTrackHit_.gpZ_[iTrackHit] = gpRecZ;
        storageTrackHit_.gpEta_[iTrackHit] = gpRecEta;
        storageTrackHit_.gpPhi_[iTrackHit] = gpRecPhi;
        storageTrackHit_.lpX_[iTrackHit] = lpX;
        storageTrackHit_.lpY_[iTrackHit] = lpY;
        storageTrackHit_.lpZ_[iTrackHit] = lpZ;
        iTrackHit++;
      } else
        printf("THIS CAN NOT HAPPEN\n");

      trkExtrap(detid, numTracks, iTrack, iRec, recoStart, lp, trackExtrap);
      numTracks++;

      if (debug_)
        printf("\tLocal Positon:  \tx = %2.2f\ty = %2.2f\tz = %2.2f\n", lpX, lpY, lpZ);
      if (debug_)
        printf("\tGlobal Position: \tx = %6.2f\ty = %6.2f\tz = %6.2f\teta = %4.2f\tphi = %3.2f\n",
               gpRecX,
               gpRecY,
               gpRecZ,
               gpRecEta,
               gpRecPhi);

      ++iRec;
    }
  }

  storageRecMuon_.n_ = iMuonHit;
  storageTrackHit_.n_ = iTrackHit;
  trackExtrap.n_ = numTracks;
}
//
// Deal with Re-Fitted Track with some station omitted.
//
// This should take the new track, match it to its track before refitting with the hits dumped, and extrapolate out to the
//  rec hits that were removed from the fit.
//
//

void ResidualRefitting::NewTrackMeasurements(edm::Handle<reco::TrackCollection> trackCollOrig,
                                             edm::Handle<reco::TrackCollection> trackColl,
                                             ResidualRefitting::storage_trackExtrap& trackExtrap) {
  int numTracks = 0;
  int recCounter = 0;

  for (reco::TrackCollection::const_iterator muon = trackColl->begin(); muon != trackColl->end(); muon++) {
    int iTrack = muon - trackColl->begin();

    reco::TrackRef trackref = reco::TrackRef(trackColl, iTrack);
    FreeTrajectoryState recoStart = ResidualRefitting::freeTrajStateMuon(trackref);

    int iTrackLink = MatchTrackWithRecHits(muon, trackCollOrig);
    reco::TrackRef ref = reco::TrackRef(trackCollOrig, iTrackLink);

    for (auto const& rec1 : ref->recHits()) {
      bool unbiasedRec = true;

      for (auto const& rec2 : muon->recHits()) {
        if (IsSameHit(*rec1, *rec2)) {
          unbiasedRec = false;
          break;
        }
      }
      if (!unbiasedRec)
        continue;

      DetId detid = rec1->geographicalId();

      auto mrhp = MuonTransientTrackingRecHit::specificBuild(
          theService->trackingGeometry()->idToDet(rec1->geographicalId()), rec1);

      trkExtrap(detid, numTracks, iTrackLink, recCounter, recoStart, rec1->localPosition(), trackExtrap);
      numTracks++;
    }
  }

  trackExtrap.n_ = numTracks;
}
//
// Find the original track that corresponds to the re-fitted track
//
int ResidualRefitting::MatchTrackWithRecHits(reco::TrackCollection::const_iterator trackIt,
                                             edm::Handle<reco::TrackCollection> ref) {
  if (debug_)
    printf("Matching a re-fitted track to the original track.\n");

  int TrackMatch = -1;

  for (auto const& rec : trackIt->recHits()) {
    bool foundMatch = false;
    for (reco::TrackCollection::const_iterator refIt = ref->begin(); refIt != ref->end(); refIt++) {
      int iTrackMatch = refIt - ref->begin();
      if (foundMatch && TrackMatch != iTrackMatch)
        break;
      for (auto const& recRef : refIt->recHits()) {
        if (!IsSameHit(*rec, *recRef))
          continue;

        foundMatch = true;
        TrackMatch = iTrackMatch;
        //	printf("Rec hit match for original track %d\n", iTrackMatch);
      }
    }
    if (!foundMatch) {
      printf("SOMETHING WENT WRONG! Could not match Track with original track!");
      exit(1);
    }
  }
  if (debug_)
    printf("Rec hit match for original track %d\n", TrackMatch);

  //	reco::TrackRef trackref=reco::TrackRef(ref,TrackMatch);
  return TrackMatch;
}
/*
//
// Match two tracks to see if one is a subset of the other
//

bool ResidualRefitting::TrackSubset(reco::TrackRef trackSub, reco::TrackRef trackTop) {

	
	bool matchAll = true;

	for (trackingRecHit_iterator recSub = trackSub->recHits().begin(); recSub!=trackSub->recHits().end(); recSub++) {

		bool matchSub = false;


		for (trackingRecHit_iterator recTop = trackTop->recHits().begin(); recTop!=trackTop->recHits().end(); recTop++) {
		
			if ( recSub == recTop ) matchSub = true;
			if (matchSub) break;

		}
		if (!matchSub) return false;

	}

	return matchAll;

}
*/

//
// Check to see if the rec hits are the same
//
bool ResidualRefitting::IsSameHit(TrackingRecHit const& hit1, TrackingRecHit const& hit2) {
  double lpx1 = hit1.localPosition().x();
  double lpy1 = hit1.localPosition().y();
  double lpz1 = hit1.localPosition().z();

  double lpx2 = hit2.localPosition().x();
  double lpy2 = hit2.localPosition().y();
  double lpz2 = hit2.localPosition().z();
  if (fabs(lpx1 - lpx2) > 1e-3)
    return false;
  //	printf("Match lpx...\n");
  if (fabs(lpy1 - lpy2) > 1e-3)
    return false;
  //	printf("Match lpy...\n");
  if (fabs(lpz1 - lpz2) > 1e-3)
    return false;
  //	printf("Match lpz...\n");

  return true;
}

//
// Store Tracker Rec Hits
//
void ResidualRefitting::StoreTrackerRecHits(DetId detid, const TrackerTopology* tTopo, int iTrack, int iRec) {
  int detector = -1;
  int subdetector = -1;
  int blade = -1;
  int disk = -1;
  int ladder = -1;
  int layer = -1;
  int module = -1;
  int panel = -1;
  int ring = -1;
  int side = -1;
  int wheel = -1;

  //Detector Info

  detector = detid.det();
  subdetector = detid.subdetId();

  if (detector != DetId::Tracker) {
    std::cout << "OMFG NOT THE TRACKER\n" << std::endl;
    return;
  }

  if (debug_)
    std::cout << "Tracker:: ";
  if (subdetector == ResidualRefitting::PXB) {
    layer = tTopo->pxbLayer(detid.rawId());
    ladder = tTopo->pxbLadder(detid.rawId());
    module = tTopo->pxbModule(detid.rawId());
    if (debug_)
      std::cout << "PXB"
                << "\tlayer = " << layer << "\tladder = " << ladder << "\tmodule = " << module;

  } else if (subdetector == ResidualRefitting::PXF) {
    side = tTopo->pxfSide(detid.rawId());
    disk = tTopo->pxfDisk(detid.rawId());
    blade = tTopo->pxfBlade(detid.rawId());
    panel = tTopo->pxfPanel(detid.rawId());
    module = tTopo->pxfModule(detid.rawId());
    if (debug_)
      std::cout << "PXF"
                << "\tside = " << side << "\tdisk = " << disk << "\tblade = " << blade << "\tpanel = " << panel
                << "\tmodule = " << module;

  } else if (subdetector == ResidualRefitting::TIB) {
    layer = tTopo->tibLayer(detid.rawId());
    module = tTopo->tibModule(detid.rawId());
    if (debug_)
      std::cout << "TIB"
                << "\tlayer = " << layer << "\tmodule = " << module;
  } else if (subdetector == ResidualRefitting::TID) {
    side = tTopo->tidSide(detid.rawId());
    wheel = tTopo->tidWheel(detid.rawId());
    ring = tTopo->tidRing(detid.rawId());
    if (debug_)
      std::cout << "TID"
                << "\tside = " << side << "\twheel = " << wheel << "\tring = " << ring;

  } else if (subdetector == ResidualRefitting::TOB) {
    layer = tTopo->tobLayer(detid.rawId());
    module = tTopo->tobModule(detid.rawId());
    if (debug_)
      std::cout << "TOB"
                << "\tlayer = " << layer << "\tmodule = " << module;

  } else if (subdetector == ResidualRefitting::TEC) {
    ring = tTopo->tecRing(detid.rawId());
    module = tTopo->tecModule(detid.rawId());
    if (debug_)
      std::cout << "TEC"
                << "\tring = " << ring << "\tmodule = " << module;
  }

  //Do Storage

  storageTrackHit_.muonLink_[iRec] = iTrack;
  storageTrackHit_.detector_[iRec] = detector;
  storageTrackHit_.subdetector_[iRec] = subdetector;
  storageTrackHit_.blade_[iRec] = blade;
  storageTrackHit_.disk_[iRec] = disk;
  storageTrackHit_.ladder_[iRec] = ladder;
  storageTrackHit_.layer_[iRec] = layer;
  storageTrackHit_.module_[iRec] = module;
  storageTrackHit_.panel_[iRec] = panel;
  storageTrackHit_.ring_[iRec] = ring;
  storageTrackHit_.side_[iRec] = side;
  storageTrackHit_.wheel_[iRec] = wheel;
}

//
// Store Muon info on P, Pt, eta, phi
//
void ResidualRefitting::muonInfo(ResidualRefitting::storage_muon& storeMuon, reco::TrackRef muon, int val) {
  storeMuon.pt_[val] = muon->pt();
  storeMuon.p_[val] = muon->p();
  storeMuon.eta_[val] = muon->eta();
  storeMuon.phi_[val] = muon->phi();
  storeMuon.charge_[val] = muon->charge();
  storeMuon.numRecHits_[val] = muon->numberOfValidHits();
  storeMuon.chiSq_[val] = muon->chi2();
  storeMuon.ndf_[val] = muon->ndof();
  storeMuon.chiSqOvrNdf_[val] = muon->normalizedChi2();
}
//
// Fill a track extrapolation
//
void ResidualRefitting::trkExtrap(const DetId& detid,
                                  int iTrk,
                                  int iTrkLink,
                                  int iRec,
                                  const FreeTrajectoryState& freeTrajState,
                                  const LocalPoint& recPoint,
                                  storage_trackExtrap& storeTemp) {
  bool dump_ = debug_;

  if (dump_)
    std::cout << "In the trkExtrap function" << std::endl;

  float gpExtrapX = -99999;
  float gpExtrapY = -99999;
  float gpExtrapZ = -99999;
  float gpExtrapEta = -99999;
  float gpExtrapPhi = -99999;

  float lpX = -99999;
  float lpY = -99999;
  float lpZ = -99999;

  //
  // Get the local positions for the recHits
  //

  float recLpX = recPoint.x();
  float recLpY = recPoint.y();
  float recLpZ = recPoint.z();

  float resX = -9999;
  float resY = -9999;
  float resZ = -9999;

  const GeomDet* gdet = theService->trackingGeometry()->idToDet(detid);

  //	TrajectoryStateOnSurface surfTest =  prop.propagate(freeTrajState, gdet->surface());
  TrajectoryStateOnSurface surfTest = thePropagator->propagate(freeTrajState, gdet->surface());

  if (surfTest.isValid()) {
    GlobalPoint globTest = surfTest.globalPosition();
    gpExtrapX = globTest.x();
    gpExtrapY = globTest.y();
    gpExtrapZ = globTest.z();
    gpExtrapEta = globTest.eta();
    gpExtrapPhi = globTest.phi();
    LocalPoint loc = surfTest.localPosition();
    if (detid.det() == DetId::Muon || detid.det() == DetId::Tracker) {
      lpX = loc.x();
      lpY = loc.y();
      lpZ = loc.z();

      resX = lpX - recLpX;
      resY = lpY - recLpY;
      resZ = lpZ - recLpZ;
    }
  }
  storeTemp.muonLink_[iTrk] = iTrkLink;
  storeTemp.recLink_[iTrk] = iRec;
  storeTemp.gpX_[iTrk] = gpExtrapX;
  storeTemp.gpY_[iTrk] = gpExtrapY;
  storeTemp.gpZ_[iTrk] = gpExtrapZ;
  storeTemp.gpEta_[iTrk] = gpExtrapEta;
  storeTemp.gpPhi_[iTrk] = gpExtrapPhi;
  storeTemp.lpX_[iTrk] = lpX;
  storeTemp.lpY_[iTrk] = lpY;
  storeTemp.lpZ_[iTrk] = lpZ;
  storeTemp.resX_[iTrk] = resX;
  storeTemp.resY_[iTrk] = resY;
  storeTemp.resZ_[iTrk] = resZ;

  printf("station: %d\tsector: %d\tresX storage: %4.2f\n", ReturnStation(detid), ReturnSector(detid), resX);
}
//
// Return the station
//
int ResidualRefitting::ReturnStation(DetId detid) {
  int station = -999;

  if (detid.det() == DetId::Muon) {
    int systemMuon = detid.subdetId();  // 1 DT; 2 CSC; 3 RPC
    if (systemMuon == MuonSubdetId::CSC) {
      CSCDetId id(detid.rawId());
      station = id.station();

    } else if (systemMuon == MuonSubdetId::DT) {
      DTWireId id(detid.rawId());
      station = id.station();

    } else if (systemMuon == MuonSubdetId::RPC) {
      RPCDetId id(detid.rawId());
      station = id.station();
    }
  }

  return station;
}
//
// Return the sector
//
int ResidualRefitting::ReturnSector(DetId detid) {
  int sector = -999;

  if (detid.det() == DetId::Muon) {
    int systemMuon = detid.subdetId();  // 1 DT; 2 CSC; 3 RPC
    if (systemMuon == MuonSubdetId::DT) {
      DTWireId id(detid.rawId());
      sector = id.sector();
    }
  }

  return sector;
}

//
// Store the SAM and Track position info at a particular rho
//
void ResidualRefitting::cylExtrapTrkSam(int recNum,
                                        reco::TrackRef track,
                                        ResidualRefitting::storage_trackExtrap& storage,
                                        double rho) {
  Cylinder::PositionType pos(0, 0, 0);
  Cylinder::RotationType rot;

  Cylinder::CylinderPointer myCylinder = Cylinder::build(pos, rot, rho);
  //	SteppingHelixPropagator inwardProp  ( theField, oppositeToMomentum );
  //	SteppingHelixPropagator outwardProp ( theField, alongMomentum );
  FreeTrajectoryState recoStart = freeTrajStateMuon(track);
  //	TrajectoryStateOnSurface recoProp = outwardProp.propagate(recoStart, *myCylinder);
  TrajectoryStateOnSurface recoProp = thePropagator->propagate(recoStart, *myCylinder);

  double xVal = -9999;
  double yVal = -9999;
  double zVal = -9999;
  double phiVal = -9999;
  double etaVal = -9999;

  if (recoProp.isValid()) {
    GlobalPoint recoPoint = recoProp.globalPosition();
    xVal = recoPoint.x();
    yVal = recoPoint.y();
    zVal = recoPoint.z();
    phiVal = recoPoint.phi();
    etaVal = recoPoint.eta();
  }
  storage.muonLink_[recNum] = recNum;
  storage.gpX_[recNum] = xVal;
  storage.gpY_[recNum] = yVal;
  storage.gpZ_[recNum] = zVal;
  storage.gpEta_[recNum] = etaVal;
  storage.gpPhi_[recNum] = phiVal;

  float rhoVal = sqrt(xVal * xVal + yVal * yVal);

  printf("Cylinder: rho = %4.2f\tphi = %4.2f\teta = %4.2f\n", rhoVal, phiVal, etaVal);
  if (debug_)
    printf("Cylinder: rho = %4.2f\tphi = %4.2f\teta = %4.2f\n", rhoVal, phiVal, etaVal);
}
///////////////////////////////////////////////////////////////////////////////
//Pre-Job junk
///////////////////////////////////////////////////////////////////////////////

//
// zero storage
//
void ResidualRefitting::zero_storage() {
  if (debug_)
    printf("zero_storage\n");

  zero_muon(&storageGmrOld_);
  zero_muon(&storageGmrNew_);
  zero_muon(&storageSamNew_);
  zero_muon(&storageTrkNew_);
  zero_muon(&storageGmrNoSt1_);
  zero_muon(&storageSamNoSt1_);
  zero_muon(&storageGmrNoSt2_);
  zero_muon(&storageSamNoSt2_);
  zero_muon(&storageGmrNoSt3_);
  zero_muon(&storageSamNoSt3_);
  zero_muon(&storageGmrNoSt4_);
  zero_muon(&storageSamNoSt4_);
  //zero out the tracker
  zero_muon(&storageGmrNoPXBLayer1);
  zero_muon(&storageGmrNoPXBLayer2);
  zero_muon(&storageGmrNoPXBLayer3);

  zero_muon(&storageGmrNoPXF);

  zero_muon(&storageGmrNoTIBLayer1);
  zero_muon(&storageGmrNoTIBLayer2);
  zero_muon(&storageGmrNoTIBLayer3);
  zero_muon(&storageGmrNoTIBLayer4);

  zero_muon(&storageGmrNoTID);

  zero_muon(&storageGmrNoTOBLayer1);
  zero_muon(&storageGmrNoTOBLayer2);
  zero_muon(&storageGmrNoTOBLayer3);
  zero_muon(&storageGmrNoTOBLayer4);
  zero_muon(&storageGmrNoTOBLayer5);
  zero_muon(&storageGmrNoTOBLayer6);

  zero_muon(&storageGmrNoTEC);

  zero_muon(&storageTrkNoPXBLayer1);
  zero_muon(&storageTrkNoPXBLayer2);
  zero_muon(&storageTrkNoPXBLayer3);

  zero_muon(&storageTrkNoPXF);

  zero_muon(&storageTrkNoTIBLayer1);
  zero_muon(&storageTrkNoTIBLayer2);
  zero_muon(&storageTrkNoTIBLayer3);
  zero_muon(&storageTrkNoTIBLayer4);

  zero_muon(&storageTrkNoTID);

  zero_muon(&storageTrkNoTOBLayer1);
  zero_muon(&storageTrkNoTOBLayer2);
  zero_muon(&storageTrkNoTOBLayer3);
  zero_muon(&storageTrkNoTOBLayer4);
  zero_muon(&storageTrkNoTOBLayer5);
  zero_muon(&storageTrkNoTOBLayer6);

  zero_muon(&storageTrkNoTEC);

  zero_trackExtrap(&storageTrackExtrapRec_);
  zero_trackExtrap(&storageTrackExtrapTracker_);
  zero_trackExtrap(&storageTrackExtrapRecNoSt1_);
  zero_trackExtrap(&storageTrackExtrapRecNoSt2_);
  zero_trackExtrap(&storageTrackExtrapRecNoSt3_);
  zero_trackExtrap(&storageTrackExtrapRecNoSt4_);

  zero_trackExtrap(&trackExtrap120_);

  zero_trackExtrap(&samExtrap120_);

  zero_trackExtrap(&storageTrackNoPXBLayer1);
  zero_trackExtrap(&storageTrackNoPXBLayer2);
  zero_trackExtrap(&storageTrackNoPXBLayer3);

  zero_trackExtrap(&storageTrackNoPXF);

  zero_trackExtrap(&storageTrackNoTIBLayer1);
  zero_trackExtrap(&storageTrackNoTIBLayer2);
  zero_trackExtrap(&storageTrackNoTIBLayer3);
  zero_trackExtrap(&storageTrackNoTIBLayer4);

  zero_trackExtrap(&storageTrackNoTOBLayer1);
  zero_trackExtrap(&storageTrackNoTOBLayer2);
  zero_trackExtrap(&storageTrackNoTOBLayer3);
  zero_trackExtrap(&storageTrackNoTOBLayer4);
  zero_trackExtrap(&storageTrackNoTOBLayer5);
  zero_trackExtrap(&storageTrackNoTOBLayer6);

  zero_trackExtrap(&storageTrackNoTEC);

  zero_trackExtrap(&storageTrackNoTID);

  storageRecMuon_.n_ = 0;
  storageTrackHit_.n_ = 0;
}
//
// Zero out a muon reference
//
void ResidualRefitting::zero_muon(ResidualRefitting::storage_muon* str) {
  str->n_ = 0;

  for (int i = 0; i < ResidualRefitting::N_MAX_STORED; i++) {
    str->pt_[i] = -9999;
    str->eta_[i] = -9999;
    str->p_[i] = -9999;
    str->phi_[i] = -9999;
    str->numRecHits_[i] = -9999;
    str->chiSq_[i] = -9999;
    str->ndf_[i] = -9999;
    str->chiSqOvrNdf_[i] = -9999;
  }
}
//
// Zero track extrapolation
//
void ResidualRefitting::zero_trackExtrap(ResidualRefitting::storage_trackExtrap* str) {
  str->n_ = 0;
  for (int i = 0; i < ResidualRefitting::N_MAX_STORED_HIT; i++) {
    str->muonLink_[i] = -9999;
    str->recLink_[i] = -9999;
    str->gpX_[i] = -9999;
    str->gpY_[i] = -9999;
    str->gpZ_[i] = -9999;
    str->gpEta_[i] = -9999;
    str->gpPhi_[i] = -9999;
    str->lpX_[i] = -9999;
    str->lpY_[i] = -9999;
    str->lpZ_[i] = -9999;
    str->resX_[i] = -9999;
    str->resY_[i] = -9999;
    str->resZ_[i] = -9999;
  }
}
//
// Begin Job
//
void ResidualRefitting::beginJob() {
  std::cout << "Creating file " << outputFileName_.c_str() << std::endl;

  outputFile_ = new TFile(outputFileName_.c_str(), "RECREATE");

  outputTree_ = new TTree("outputTree", "outputTree");

  outputTree_->Branch("eventInfo",
                      &eventInfo_,
                      "evtNum_/I:"
                      "runNum_/I");

  ResidualRefitting::branchMuon(storageGmrOld_, "gmrOld");
  ResidualRefitting::branchMuon(storageGmrNew_, "gmrNew");
  ResidualRefitting::branchMuon(storageGmrNoSt1_, "gmrNoSt1");
  ResidualRefitting::branchMuon(storageGmrNoSt2_, "gmrNoSt2");
  ResidualRefitting::branchMuon(storageGmrNoSt3_, "gmrNoSt3");
  ResidualRefitting::branchMuon(storageGmrNoSt4_, "gmrNoSt4");

  ResidualRefitting::branchMuon(storageSamNew_, "samNew");
  ResidualRefitting::branchMuon(storageSamNoSt1_, "samNoSt1");
  ResidualRefitting::branchMuon(storageSamNoSt2_, "samNoSt2");
  ResidualRefitting::branchMuon(storageSamNoSt3_, "samNoSt3");
  ResidualRefitting::branchMuon(storageSamNoSt4_, "samNoSt4");

  ResidualRefitting::branchMuon(storageTrkNew_, "trkNew");
  ResidualRefitting::branchMuon(storageGmrNoPXBLayer1, "gmrNoPXBLayer1");
  ResidualRefitting::branchMuon(storageGmrNoPXBLayer2, "gmrNoPXBLayer2");
  ResidualRefitting::branchMuon(storageGmrNoPXBLayer3, "gmrNoPXBLayer3");
  ResidualRefitting::branchMuon(storageGmrNoPXF, "gmrNoPXF");
  ResidualRefitting::branchMuon(storageGmrNoTIBLayer1, "gmrNoTIBLayer1");
  ResidualRefitting::branchMuon(storageGmrNoTIBLayer2, "gmrNoTIBLayer2");
  ResidualRefitting::branchMuon(storageGmrNoTIBLayer3, "gmrNoTIBLayer3");
  ResidualRefitting::branchMuon(storageGmrNoTIBLayer4, "gmrNoTIBLayer4");
  ResidualRefitting::branchMuon(storageGmrNoTID, "gmrNoTID");
  ResidualRefitting::branchMuon(storageGmrNoTOBLayer1, "gmrNoTOBLayer1");
  ResidualRefitting::branchMuon(storageGmrNoTOBLayer2, "gmrNoTOBLayer2");
  ResidualRefitting::branchMuon(storageGmrNoTOBLayer3, "gmrNoTOBLayer3");
  ResidualRefitting::branchMuon(storageGmrNoTOBLayer4, "gmrNoTOBLayer4");
  ResidualRefitting::branchMuon(storageGmrNoTOBLayer5, "gmrNoTOBLayer5");
  ResidualRefitting::branchMuon(storageGmrNoTOBLayer6, "gmrNoTOBLayer6");
  ResidualRefitting::branchMuon(storageGmrNoTEC, "gmrNoTEC");

  ResidualRefitting::branchMuon(storageTrkNoPXBLayer1, "trkNoPXBLayer1");
  ResidualRefitting::branchMuon(storageTrkNoPXBLayer2, "trkNoPXBLayer2");
  ResidualRefitting::branchMuon(storageTrkNoPXBLayer3, "trkNoPXBLayer3");
  ResidualRefitting::branchMuon(storageTrkNoPXF, "trkNoPXF");
  ResidualRefitting::branchMuon(storageTrkNoTIBLayer1, "trkNoTIBLayer1");
  ResidualRefitting::branchMuon(storageTrkNoTIBLayer2, "trkNoTIBLayer2");
  ResidualRefitting::branchMuon(storageTrkNoTIBLayer3, "trkNoTIBLayer3");
  ResidualRefitting::branchMuon(storageTrkNoTIBLayer4, "trkNoTIBLayer4");
  ResidualRefitting::branchMuon(storageTrkNoTID, "trkNoTID");
  ResidualRefitting::branchMuon(storageTrkNoTOBLayer1, "trkNoTOBLayer1");
  ResidualRefitting::branchMuon(storageTrkNoTOBLayer2, "trkNoTOBLayer2");
  ResidualRefitting::branchMuon(storageTrkNoTOBLayer3, "trkNoTOBLayer3");
  ResidualRefitting::branchMuon(storageTrkNoTOBLayer4, "trkNoTOBLayer4");
  ResidualRefitting::branchMuon(storageTrkNoTOBLayer5, "trkNoTOBLayer5");
  ResidualRefitting::branchMuon(storageTrkNoTOBLayer6, "trkNoTOBLayer6");
  ResidualRefitting::branchMuon(storageTrkNoTEC, "trkNoTEC");

  outputBranch_ = outputTree_->Branch("recHitsNew",
                                      &storageRecMuon_,

                                      "n_/I:"
                                      "muonLink_[1000]/I:"

                                      "system_[1000]/I:"
                                      "endcap_[1000]/I:"
                                      "station_[1000]/I:"
                                      "ring_[1000]/I:"
                                      "chamber_[1000]/I:"
                                      "layer_[1000]/I:"
                                      "superLayer_[1000]/I:"
                                      "wheel_[1000]/I:"
                                      "sector_[1000]/I:"

                                      "gpX_[1000]/F:"
                                      "gpY_[1000]/F:"
                                      "gpZ_[1000]/F:"
                                      "gpEta_[1000]/F:"
                                      "gpPhi_[1000]/F:"
                                      "lpX_[1000]/F:"
                                      "lpY_[1000]/F:"
                                      "lpZ_[1000]/F");

  outputBranch_ = outputTree_->Branch("recHitsTracker",
                                      &storageTrackHit_,

                                      "n_/I:"

                                      "muonLink_[1000]/I:"
                                      "detector_[1000]/I:"
                                      "subdetector_[1000]/I:"
                                      "blade_[1000]/I:"
                                      "disk_[1000]/I:"
                                      "ladder_[1000]/I:"
                                      "layer_[1000]/I:"
                                      "module_[1000]/I:"
                                      "panel_[1000]/I:"
                                      "ring_[1000]/I:"
                                      "side_[1000]/I:"
                                      "wheel_[1000]/I:"

                                      "gpX_[1000]/F:"
                                      "gpY_[1000]/F:"
                                      "gpZ_[1000]/F:"
                                      "gpEta_[1000]/F:"
                                      "gpPhi_[1000]/F:"
                                      "lpX_[1000]/F:"
                                      "lpY_[1000]/F:"
                                      "lpZ_[1000]/F");

  ResidualRefitting::branchTrackExtrap(storageTrackExtrapRec_, "trkExtrap");
  ResidualRefitting::branchTrackExtrap(storageTrackExtrapRecNoSt1_, "trkExtrapNoSt1");
  ResidualRefitting::branchTrackExtrap(storageTrackExtrapRecNoSt2_, "trkExtrapNoSt2");
  ResidualRefitting::branchTrackExtrap(storageTrackExtrapRecNoSt3_, "trkExtrapNoSt3");
  ResidualRefitting::branchTrackExtrap(storageTrackExtrapRecNoSt4_, "trkExtrapNoSt4");

  ResidualRefitting::branchTrackExtrap(storageTrackExtrapTracker_, "trkExtrapTracker");
  ResidualRefitting::branchTrackExtrap(storageTrackNoPXF, "trkExtrapNoPXF");
  ResidualRefitting::branchTrackExtrap(storageTrackNoPXBLayer1, "trkExtrapNoPXBLayer1");
  ResidualRefitting::branchTrackExtrap(storageTrackNoPXBLayer2, "trkExtrapNoPXBLayer2");
  ResidualRefitting::branchTrackExtrap(storageTrackNoPXBLayer3, "trkExtrapNoPXBLayer3");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTIBLayer1, "trkExtrapNoTIBLayer1");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTIBLayer2, "trkExtrapNoTIBLayer2");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTIBLayer3, "trkExtrapNoTIBLayer3");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTIBLayer4, "trkExtrapNoTIBLayer4");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTID, "trkExtrapNoTID");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer1, "trkExtrapNoTOBLayer1");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer2, "trkExtrapNoTOBLayer2");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer3, "trkExtrapNoTOBLayer3");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer4, "trkExtrapNoTOBLayer4");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer5, "trkExtrapNoTOBLayer5");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTOBLayer6, "trkExtrapNoTOBLayer6");
  ResidualRefitting::branchTrackExtrap(storageTrackNoTEC, "trkExtrapNoTEC");

  ResidualRefitting::branchTrackExtrap(trackExtrap120_, "trackCyl120");
  ResidualRefitting::branchTrackExtrap(samExtrap120_, "samCyl120");
}
//
// Set the Muon Branches
//
void ResidualRefitting::branchMuon(ResidualRefitting::storage_muon& storageTmp, std::string branchName) {
  outputBranch_ = outputTree_->Branch(branchName.c_str(),
                                      &storageTmp,
                                      "n_/I:"
                                      "charge_[10]/I:"
                                      "pt_[10]/F:"
                                      "eta_[10]/F:"
                                      "p_[10]/F:"
                                      "phi_[10]/F:"
                                      "numRecHits_[10]/I:"
                                      "chiSq_[10]/F:"
                                      "ndf_[10]/F:"
                                      "chiSqOvrNdf_[10]/F"

  );
}
//
// Set the Branches for Track Extrapolations
//
void ResidualRefitting::branchTrackExtrap(ResidualRefitting::storage_trackExtrap& storageTmp, std::string branchName) {
  outputBranch_ = outputTree_->Branch(branchName.c_str(),
                                      &storageTmp,
                                      "n_/I:"
                                      "muonLink_[1000]/I:"
                                      "recLink_[1000]/I:"
                                      "gpX_[1000]/F:"
                                      "gpY_[1000]/F:"
                                      "gpZ_[1000]/F:"
                                      "gpEta_[1000]/F:"
                                      "gpPhi_[1000]/F:"
                                      "lpX_[1000]/F:"
                                      "lpY_[1000]/F:"
                                      "lpZ_[1000]/F:"
                                      "resX_[1000]/F:"
                                      "resY_[1000]/F:"
                                      "resZ_[1000]/F"

  );
}
//
// End Job
//
void ResidualRefitting::endJob() {
  outputFile_->Write();

  outputFile_->Close();
}
//
// Return a Free Trajectory state for a muon track
//
FreeTrajectoryState ResidualRefitting::freeTrajStateMuon(reco::TrackRef muon) {
  math::XYZPoint innerPos = muon->referencePoint();
  math::XYZVector innerMom = muon->momentum();
  if (debug_)
    std::cout << "Inner Pos: "
              << "\tx = " << innerPos.X() << "\ty = " << innerPos.Y() << "\tz = " << innerPos.Z() << std::endl;

  GlobalPoint innerPoint(innerPos.X(), innerPos.Y(), innerPos.Z());
  GlobalVector innerVec(innerMom.X(), innerMom.Y(), innerMom.Z());

  FreeTrajectoryState recoStart(innerPoint, innerVec, muon->charge(), theField);
  return recoStart;
}

/////////////////////////////////////////////////////////////////////////////////
// nTuple value Dumps
/////////////////////////////////////////////////////////////////////////////////

//
// dump Track Extrapolation
//
void ResidualRefitting::dumpTrackExtrap(const ResidualRefitting::storage_trackExtrap& track) {
  std::cout << "\n\nExtrapolation Dump:\n";
  for (unsigned int i = 0; i < (unsigned int)track.n_; i++) {
    //		double rho = sqrt( (float)track.gpX_[i] * (float)track.gpX_[i] + (float)track.gpY_[i] * (float)track.gpY_[i]  );

    printf("%d\tmuonLink= %d", i, (int)track.muonLink_[i]);
    printf("\trecLink = %d", (int)track.recLink_[i]);
    //		printf ("\tGlobal\tx = %0.3f"		,		(float)track.gpX_[i]	);
    //		printf ("\ty = %0.3f"		,		(float)track.gpY_[i]	);
    //		printf ("\tz = %0.3f"		,		(float)track.gpZ_[i]	);
    //		printf ("\trho =%0.3f"		,		rho						);
    //		printf ("\teta = %0.3f"		,		(float)track.gpEta_[i]	);
    //		printf ("\tphi = %0.3f"		,		(float)track.gpPhi_[i]	);
    printf("\t\tLocal\tx = %0.3f", (float)track.lpX_[i]);
    printf("\ty = %0.3f", (float)track.lpY_[i]);
    printf("\tz = %0.3f\n", (float)track.lpZ_[i]);
  }
}
//
// dump Muon Rec Hits
//
void ResidualRefitting::dumpMuonRecHits(const ResidualRefitting::storage_hit& hit) {
  std::cout << "Muon Rec Hits Dump:\n";
  for (unsigned int i = 0; i < (unsigned int)hit.n_; i++) {
    //		double rho = sqrt( (float)hit.gpX_[i] * (float)hit.gpX_[i] + (float)hit.gpY_[i] * (float)hit.gpY_[i]  );

    printf("%d\tsubdetector = %d\t superLayer =%d", i, (int)hit.system_[i], (int)hit.superLayer_[i]);
    //		printf ("\tGlobal\tx = %0.3f"			,		(float)hit.gpX_[i]			);
    //		printf ("\ty = %0.3f"				,		(float)hit.gpY_[i]			);
    //		printf ("\tz = %0.3f"				,		(float)hit.gpZ_[i]			);
    //		printf ("\trho =%0.3f"				,		rho							);
    //		printf ("\teta = %0.3f"				,		(float)hit.gpEta_[i]		);
    //		printf ("\tphi = %0.3f\n"			,		(float)hit.gpPhi_[i]		);
    printf("\t\tLocal\tx = %0.3f", (float)hit.lpX_[i]);
    printf("\ty = %0.3f", (float)hit.lpY_[i]);
    printf("\tz = %0.3f\n", (float)hit.lpZ_[i]);
  }
}
//
// dump Tracker Rec Hits
//
void ResidualRefitting::dumpTrackHits(const ResidualRefitting::storage_trackHit& hit) {
  std::cout << "Tracker Rec Hits Dump:\n";
  for (unsigned int i = 0; i < (unsigned int)hit.n_; i++) {
    //		double rho = sqrt( (float)hit.gpX_[i] * (float)hit.gpX_[i] + (float)hit.gpY_[i] * (float)hit.gpY_[i]  );

    printf("%d\tsubdetector = %d", i, (int)hit.subdetector_[i]);
    printf("\tlayer = %d", (int)hit.layer_[i]);
    //		printf ("\tGlobal\tx = %0.3f"			,		(float)hit.gpX_[i]			);
    //		printf ("\ty = %0.3f"				,		(float)hit.gpY_[i]			);
    //		printf ("\tz = %0.3f"				,		(float)hit.gpZ_[i]			);
    //		printf ("\trho =%0.3f"				,		rho							);
    //		printf ("\teta = %0.3f"				,		(float)hit.gpEta_[i]		);
    //		printf ("\tphi = %0.3f\n"			,		(float)hit.gpPhi_[i]		);
    printf("\t\tLocal\tx = %0.3f", (float)hit.lpX_[i]);
    printf("\ty = %0.3f", (float)hit.lpY_[i]);
    printf("\tz = %0.3f\n", (float)hit.lpZ_[i]);
  }
}
//
//Dump a TrackRef
//
void ResidualRefitting::dumpTrackRef(reco::TrackRef muon, std::string str) {
  float pt = muon->pt();
  float p = muon->p();
  float eta = muon->eta();
  float phi = muon->phi();
  printf("\t%s: \tp = %4.2f \t pt = %4.2f \t eta = %4.2f \t phi = %4.2f\n", str.c_str(), p, pt, eta, phi);
}

DEFINE_FWK_MODULE(ResidualRefitting);

////////////////////////////////////////////////////////////////////////////////////////////
//Deprecated
////////////////////////////////////////////////////////////////////////////////////////////
