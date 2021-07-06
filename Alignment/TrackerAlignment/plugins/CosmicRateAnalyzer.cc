// -*- C++ -*-
//
// Package:    CosmicTrackTool/CosmicRateAnalyzer
// Class:      CosmicRateAnalyzer
//
/**\class CosmicRateAnalyzer CosmicRateAnalyzer.cc CosmicTrackTool/CosmicRateAnalyzer/plugins/CosmicRateAnalyzer.cc

 Description :
  This Analyzer creates tuple, having necessary infromation for Cosmic Track Rate and Event Rate calculations.
  Tuples created by this analyzer also have some kinematic information. This tuple is input to some offline
  macros that make Rate plots and Kinematical plots.

Implementation : Documentation for running this tool is described in twiki :
https://twiki.cern.ch/twiki/bin/view/CMS/TkAlCosmicsRateMonitoring

*/
// Originally created:  Justyna Magdalena Tomaszewska,,,
// Revisited by: Ashutosh Bhardwaj and Kirti Ranjan
// Further Developed by: Sumit Keshri (sumit.keshri@cern.ch) & Saumya (saumya.saumya@cern.ch)
//
//         Created:  Sat, 30 May 2015 20:14:35 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "DataFormats/Provenance/interface/Timestamp.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <TTree.h>

//
// class declaration
//
class CosmicRateAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit CosmicRateAnalyzer(const edm::ParameterSet&);
  ~CosmicRateAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  static double stampToReal(edm::Timestamp time) { return time.unixTime() + time.microsecondOffset() * 1e-6; }
  void ClearInEventLoop();
  void ClearInEndRun();
  // ----------member data ---------------------------
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<SiStripLatency, SiStripLatencyRcd> latencyToken_;
  edm::EDGetTokenT<reco::TrackCollection> trackTags_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clustercollectionToken_;
  edm::EDGetTokenT<reco::MuonCollection> muonTags_;
  edm::RunNumber_t lastrunnum;
  double lastruntime, magField;
  edm::Service<TFileService> fs;

  unsigned int DetectorID;
  TTree* treeEvent;
  TTree* treeRun;
  TTree* treeCluster;

  //---------  Temporary varibles to store the values till the Run tree is filled  ------//
  // a) For track rates
  int events;
  int track_BPIX;
  int track_FPIX;
  int track_PIXEL;
  int track_TEC;
  int track_TECM;
  int track_TECP;
  int track_TOB;
  int track_TIB;
  int track_TID;
  int track_TIDM;
  int track_TIDP;

  // b) For PIXEL Hit Rates by layers
  int hit_Total;
  int hit_PIX;
  int hit_BPIX;
  int hit_BPIX_layer1;
  int hit_BPIX_layer2;
  int hit_BPIX_layer3;
  int hit_BPIX_layer4;
  int hit_FPIX;
  int hit_FPIX_disk1;
  int hit_FPIX_disk2;
  int hit_FPIX_disk3;
  int hit_FPIX_disk1_plus;
  int hit_FPIX_disk2_plus;
  int hit_FPIX_disk3_plus;
  int hit_FPIX_disk1_minus;
  int hit_FPIX_disk2_minus;
  int hit_FPIX_disk3_minus;

  std::vector<int> v_ntrk;
  int ntrk;
  int ntrk_runnum;

  //---------- Branch Variables in tree Run ----------//
  // a) Track Rate
  int number_of_tracks;
  int number_of_tracks_PIX;
  int number_of_tracks_FPIX;
  int number_of_tracks_BPIX;
  int number_of_tracks_TEC;
  int number_of_tracks_TECP;
  int number_of_tracks_TECM;
  int number_of_tracks_TOB;
  int number_of_tracks_TIB;
  int number_of_tracks_TID;
  int number_of_tracks_TIDP;
  int number_of_tracks_TIDM;
  int number_of_events;
  edm::RunNumber_t runnum;
  double run_time;

  // b) For Hit Rate per PIXEL layer
  int number_of_hits_Total;
  int number_of_hits_PIX;
  int number_of_hits_BPIX;
  int number_of_hits_BPIX_layer1;
  int number_of_hits_BPIX_layer2;
  int number_of_hits_BPIX_layer3;
  int number_of_hits_BPIX_layer4;
  int number_of_hits_FPIX;
  int number_of_hits_FPIX_disk1;
  int number_of_hits_FPIX_disk2;
  int number_of_hits_FPIX_disk3;
  int number_of_hits_FPIX_disk1_plus;
  int number_of_hits_FPIX_disk2_plus;
  int number_of_hits_FPIX_disk3_plus;
  int number_of_hits_FPIX_disk1_minus;
  int number_of_hits_FPIX_disk2_minus;
  int number_of_hits_FPIX_disk3_minus;

  //---------- Branch Variables in tree Event: Track parameters ----------//
  std::vector<double> pt;
  std::vector<double> charge;
  std::vector<double> chi2;
  std::vector<double> chi2_ndof;
  std::vector<double> eta;
  std::vector<double> theta;
  std::vector<double> phi;
  std::vector<double> p;
  std::vector<double> d0;
  std::vector<double> dz;
  std::vector<double> nvh;
  std::vector<double> DTtime;
  std::vector<int> nh_PIXEL;
  std::vector<int> nh_BPIX;
  std::vector<int> nh_FPIX;
  std::vector<int> nh_TIB;
  std::vector<int> nh_TOB;
  std::vector<int> nh_TID;
  std::vector<int> nh_TEC;

  //------ Temporary variables to store Hits per track till the Event tree is filled -------//
  //FPIX+/-, BPIX+/-,TEC+/-, TID+/- can also be added similar way in case required
  int nHits_PIXEL;

  //------ Variables to keep track of total events and tracks ------//
  int nTotalTracks, nTotalEvents;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CosmicRateAnalyzer::CosmicRateAnalyzer(const edm::ParameterSet& iConfig)
    : magFieldToken_(esConsumes()),
      topoToken_(esConsumes()),
      latencyToken_(esConsumes()),
      trackTags_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksInputTag"))),
      clustercollectionToken_(
          consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("tracksInputTag"))),
      muonTags_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muonsInputTag"))) {
  //now do what ever initialization is needed
  //
  usesResource(TFileService::kSharedResource);
  treeEvent = fs->make<TTree>("Event", "");
  treeRun = fs->make<TTree>("Run", "");
  treeCluster = fs->make<TTree>("Cluster", "");

  nTotalTracks = 0;
  nTotalEvents = 0;
}

CosmicRateAnalyzer::~CosmicRateAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//
void CosmicRateAnalyzer::ClearInEventLoop() {
  pt.clear();
  charge.clear();
  chi2.clear();
  chi2_ndof.clear();
  eta.clear();
  theta.clear();
  phi.clear();
  p.clear();
  d0.clear();
  dz.clear();
  nvh.clear();
  DTtime.clear();
  nh_PIXEL.clear();
  nh_BPIX.clear();
  nh_FPIX.clear();
  nh_TIB.clear();
  nh_TOB.clear();
  nh_TID.clear();
  nh_TEC.clear();
}

// ------------ method called for each event  ------------
void CosmicRateAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  using reco::TrackCollection;
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(trackTags_, tracks);

  const TrackerTopology* const tTopo = &iSetup.getData(topoToken_);
  const MagneticField* magneticField = &iSetup.getData(magFieldToken_);
  magField = magneticField->inTesla(GlobalPoint(0, 0, 0)).mag();
  //const SiStripLatency* apvlat = &iSetup.getData(latencyToken_); // unused (for the moment)

  edm::Timestamp ts_begin = iEvent.getRun().beginTime();
  double t_begin = stampToReal(ts_begin);
  edm::Timestamp ts_end = iEvent.getRun().endTime();
  double t_end = stampToReal(ts_end);

  lastruntime = t_end - t_begin;
  lastrunnum = iEvent.getRun().run();

  if (!tracks->empty())
    v_ntrk.push_back(tracks->size());

  ntrk = 0;
  for (TrackCollection::const_iterator itTrack1 = tracks->begin(); itTrack1 != tracks->end(); ++itTrack1) {
    pt.push_back(itTrack1->pt());
    charge.push_back(itTrack1->charge());
    chi2.push_back(itTrack1->chi2());
    chi2_ndof.push_back(itTrack1->normalizedChi2());
    eta.push_back(itTrack1->eta());
    theta.push_back(itTrack1->theta());
    phi.push_back(itTrack1->phi());
    p.push_back(itTrack1->p());
    d0.push_back(itTrack1->d0());
    dz.push_back(itTrack1->dz());
    nvh.push_back(itTrack1->numberOfValidHits());
    nh_BPIX.push_back(itTrack1->hitPattern().numberOfValidPixelBarrelHits());
    nh_FPIX.push_back(itTrack1->hitPattern().numberOfValidPixelEndcapHits());
    nh_TIB.push_back(itTrack1->hitPattern().numberOfValidStripTIBHits());
    nh_TOB.push_back(itTrack1->hitPattern().numberOfValidStripTOBHits());
    nh_TID.push_back(itTrack1->hitPattern().numberOfValidStripTIDHits());
    nh_TEC.push_back(itTrack1->hitPattern().numberOfValidStripTECHits());

    nHits_PIXEL = 0;

    int nhitinBPIX = 0;
    int nhitinFPIX = 0;
    int nhitinPIXEL = 0;
    int nhitinTEC = 0;
    int nhitinTOB = 0;
    int nhitinTIB = 0;
    int nhitinTID = 0;
    int nhitinTECminus = 0;
    int nhitinTECplus = 0;
    int nhitinTIDminus = 0;
    int nhitinTIDplus = 0;
    int countHit = 0;

    for (auto const& hit1 : itTrack1->recHits()) {
      const DetId detId1(hit1->geographicalId());
      const int subdetId1 = detId1.subdetId();
      uint32_t detid_db = detId1.rawId();
      if (!hit1->isValid())
        continue;  // only real hits count as in itTrack1->numberOfValidHits()
      hit_Total++;

      ///////////////////////////////////////////////////////////////////////////////////////////////////
      // 			 Hit information in PixelBarrel                          		 //
      ///////////////////////////////////////////////////////////////////////////////////////////////////
      if (PixelSubdetector::PixelBarrel == subdetId1) {
        ++nhitinBPIX;   //for cosmic track rate evaluation
        ++nhitinPIXEL;  //for cosmic track rate evaluation
        ++nHits_PIXEL;  // for PIXEL hits per track in Event Tree
        ++hit_PIX;      // for cosmic PIXEL hit rates per layer
        ++hit_BPIX;     // for cosmic PIXEL hit rates per layer

        int BPIX_layer = (tTopo->pxbLayer(detid_db));
        if (BPIX_layer == 1) {
          ++hit_BPIX_layer1;  // for cosmic PIXEL hit rates per layer
        } else if (BPIX_layer == 2) {
          ++hit_BPIX_layer2;  // for cosmic PIXEL hit rates per layer
        } else if (BPIX_layer == 3) {
          ++hit_BPIX_layer3;  // for cosmic PIXEL hit rates per layer
        } else if (BPIX_layer == 4) {
          ++hit_BPIX_layer4;  // for cosmic PIXEL hit rates per layer
        } else {
          std::cout << "CAUTION : Check Phase! BPIX layer not in {1,2,3}!" << std::endl;
          std::cout << "Layer is : " << BPIX_layer << std::endl;
        }

      }
      ///////////////////////////////////////////////////////////////////////////////////////////////////
      //			Hit information in PixelEndcap                                  	//
      //////////////////////////////////////////////////////////////////////////////////////////////////
      else if (PixelSubdetector::PixelEndcap == subdetId1) {
        ++nhitinFPIX;
        ++nhitinPIXEL;

        ++nHits_PIXEL;
        ++hit_PIX;   // for cosmic PIXEL hit rates per layer
        ++hit_FPIX;  // for cosmic PIXEL hit rates per layer

        int FPIX_side = (tTopo->pxfSide(detid_db));
        int FPIX_disk = (tTopo->pxfDisk(detid_db));

        if (FPIX_disk == 1) {
          ++hit_FPIX_disk1;  // for cosmic PIXEL hit rates per layer
        } else if (FPIX_disk == 2) {
          ++hit_FPIX_disk2;  // for cosmic PIXEL hit rates per layer
        } else if (FPIX_disk == 3) {
          ++hit_FPIX_disk3;  // for cosmic PIXEL hit rates per layer
        } else {
          std::cout << "CAUTION : Check Phase! FPIX disk not in {1,2}!" << std::endl;
          std::cout << "Disk is : " << FPIX_disk << std::endl;
        }

        if (FPIX_disk == 1 && FPIX_side == 1) {
          ++hit_FPIX_disk1_minus;  // for cosmic PIXEL hit rates per layer
        } else if (FPIX_disk == 1 && FPIX_side == 2) {
          ++hit_FPIX_disk1_plus;  // for cosmic PIXEL hit rates per layer
        } else if (FPIX_disk == 2 && FPIX_side == 1) {
          ++hit_FPIX_disk2_minus;  // for cosmic PIXEL hit rates per layer
        } else if (FPIX_disk == 2 && FPIX_side == 2) {
          ++hit_FPIX_disk2_plus;  // for cosmic PIXEL hit rates per layer
        } else if (FPIX_disk == 3 && FPIX_side == 1) {
          ++hit_FPIX_disk3_minus;  // for cosmic PIXEL hit rates per layer
        } else if (FPIX_disk == 3 && FPIX_side == 2) {
          ++hit_FPIX_disk3_plus;  // for cosmic PIXEL hit rates per layer
        } else {
          std::cout << "CAUTION : FPIX side not in {1,2}!" << std::endl;
        }

      }
      //////////////////////////////////////////////////////////////////////////////////////////////////
      //			Hit information in TEC							//
      //////////////////////////////////////////////////////////////////////////////////////////////////
      else if (SiStripDetId::TEC == subdetId1) {
        ++nhitinTEC;

        if (tTopo->tecIsZMinusSide(detId1)) {
          ++nhitinTECminus;
        } else {
          ++nhitinTECplus;
        }
      }
      //////////////////////////////////////////////////////////////////////////////////////////////////
      //			Hit information in TOB		   		                        //
      /////////////////////////////////////////////////////////////////////////////////////////////////
      else if (SiStripDetId::TOB == subdetId1) {
        ++nhitinTOB;
      }
      //////////////////////////////////////////////////////////////////////////////////////////////////
      //			Hit information in TIB		                                	//
      /////////////////////////////////////////////////////////////////////////////////////////////////
      else if (SiStripDetId::TIB == subdetId1) {
        ++nhitinTIB;
      }
      //////////////////////////////////////////////////////////////////////////////////////////////////
      //			Hit information in TID		                                	//
      /////////////////////////////////////////////////////////////////////////////////////////////////
      else if (SiStripDetId::TID == subdetId1) {
        ++nhitinTID;

        if (tTopo->tidIsZMinusSide(detId1)) {
          ++nhitinTIDminus;
        } else {
          ++nhitinTIDplus;
        }
      }

      countHit++;
    }  // for Loop over Hits

    nh_PIXEL.push_back(nHits_PIXEL);

    if (nhitinBPIX > 0) {
      track_BPIX++;
    }
    if (nhitinFPIX > 0) {
      track_FPIX++;
    }
    if (nhitinPIXEL > 0) {
      track_PIXEL++;
    }
    if (nhitinTEC > 0) {
      track_TEC++;
    }
    if (nhitinTECminus > 0) {
      track_TECM++;
    }
    if (nhitinTECplus > 0) {
      track_TECP++;
    }
    if (nhitinTOB > 0) {
      track_TOB++;
    }
    if (nhitinTIB > 0) {
      track_TIB++;
    }
    if (nhitinTID > 0) {
      track_TID++;
    }
    if (nhitinTIDminus > 0) {
      track_TIDM++;
    }
    if (nhitinTIDplus > 0) {
      track_TIDP++;
    }

    ntrk++;
    ntrk_runnum++;
    nTotalTracks++;
  }  // for Loop over TrackCollection
  events++;

  Handle<edmNew::DetSetVector<SiStripCluster> > cluster;
  iEvent.getByToken(clustercollectionToken_, cluster);

  for (edmNew::DetSetVector<SiStripCluster>::const_iterator det = cluster->begin(); det != cluster->end(); ++det) {
    DetectorID = (det->detId());
    treeCluster->Fill();
  }

  edm::Handle<reco::MuonCollection> muH;
  iEvent.getByToken(muonTags_, muH);
  const reco::MuonCollection& muonsT0 = *(muH.product());
  float time = -9999.;
  for (unsigned int i = 0; i < muonsT0.size(); i++) {
    //DT time
    reco::MuonTime mt0 = muonsT0[i].time();
    time = mt0.timeAtIpInOut;
    DTtime.push_back(time);
  }

  treeEvent->Fill();
  ClearInEventLoop();
  nTotalEvents++;

}  //Event Loop

// ------------ method called once each job just before starting event loop  ------------
void CosmicRateAnalyzer::beginJob() {
  //--- Event tree ---//
  treeEvent->Branch("pt", &pt);
  treeEvent->Branch("charge", &charge);
  treeEvent->Branch("chi2", &chi2);
  treeEvent->Branch("chi2_ndof", &chi2_ndof);
  treeEvent->Branch("eta", &eta);
  treeEvent->Branch("theta", &theta);
  treeEvent->Branch("phi", &phi);
  treeEvent->Branch("p", &p);
  treeEvent->Branch("d0", &d0);
  treeEvent->Branch("dz", &dz);
  treeEvent->Branch("nvh", &nvh);
  treeEvent->Branch("ntrk", &ntrk);
  treeEvent->Branch("nHitsPIXEL", &nh_PIXEL);
  treeEvent->Branch("nHitsBPIX", &nh_BPIX);
  treeEvent->Branch("nHitsFPIX", &nh_FPIX);
  treeEvent->Branch("nHitsTIB", &nh_TIB);
  treeEvent->Branch("nHitsTOB", &nh_TOB);
  treeEvent->Branch("nHitsTID", &nh_TID);
  treeEvent->Branch("nHitsTEC", &nh_TEC);
  treeEvent->Branch("DTtime", &DTtime);
  treeEvent->Branch("magField", &magField);

  //--- Run tree ---//
  treeRun->Branch("run_time", &run_time);
  treeRun->Branch("runnum", &runnum);
  // a) For track Rate Calculation
  treeRun->Branch("number_of_events", &number_of_events);
  treeRun->Branch("number_of_tracks", &number_of_tracks);
  treeRun->Branch("number_of_tracks_PIX", &number_of_tracks_PIX);
  treeRun->Branch("number_of_tracks_FPIX", &number_of_tracks_FPIX);
  treeRun->Branch("number_of_tracks_BPIX", &number_of_tracks_BPIX);
  treeRun->Branch("number_of_tracks_TID", &number_of_tracks_TID);
  treeRun->Branch("number_of_tracks_TIDM", &number_of_tracks_TIDM);
  treeRun->Branch("number_of_tracks_TIDP", &number_of_tracks_TIDP);
  treeRun->Branch("number_of_tracks_TIB", &number_of_tracks_TIB);
  treeRun->Branch("number_of_tracks_TEC", &number_of_tracks_TEC);
  treeRun->Branch("number_of_tracks_TECP", &number_of_tracks_TECP);
  treeRun->Branch("number_of_tracks_TECM", &number_of_tracks_TECM);
  treeRun->Branch("number_of_tracks_TOB", &number_of_tracks_TOB);
  // a) For PIXEL Hit Rate Calculation
  treeRun->Branch("number_of_hits_Total", &number_of_hits_Total);
  treeRun->Branch("number_of_hits_PIX", &number_of_hits_PIX);
  treeRun->Branch("number_of_hits_BPIX", &number_of_hits_BPIX);
  treeRun->Branch("number_of_hits_BPIX_layer1", &number_of_hits_BPIX_layer1);
  treeRun->Branch("number_of_hits_BPIX_layer2", &number_of_hits_BPIX_layer2);
  treeRun->Branch("number_of_hits_BPIX_layer3", &number_of_hits_BPIX_layer3);
  treeRun->Branch("number_of_hits_BPIX_layer4", &number_of_hits_BPIX_layer4);
  treeRun->Branch("number_of_hits_FPIX", &number_of_hits_FPIX);
  treeRun->Branch("number_of_hits_FPIX_disk1", &number_of_hits_FPIX_disk1);
  treeRun->Branch("number_of_hits_FPIX_disk2", &number_of_hits_FPIX_disk2);
  treeRun->Branch("number_of_hits_FPIX_disk3", &number_of_hits_FPIX_disk3);
  treeRun->Branch("number_of_hits_FPIX_disk1_plus", &number_of_hits_FPIX_disk1_plus);
  treeRun->Branch("number_of_hits_FPIX_disk1_minus", &number_of_hits_FPIX_disk1_minus);
  treeRun->Branch("number_of_hits_FPIX_disk2_plus", &number_of_hits_FPIX_disk2_plus);
  treeRun->Branch("number_of_hits_FPIX_disk2_minus", &number_of_hits_FPIX_disk2_minus);
  treeRun->Branch("number_of_hits_FPIX_disk3_plus", &number_of_hits_FPIX_disk3_plus);
  treeRun->Branch("number_of_hits_FPIX_disk3_minus", &number_of_hits_FPIX_disk3_minus);

  //--- Cluster tree ---//
  treeCluster->Branch("DetID", &DetectorID);
}

// ------------ method called once each job just after ending the event loop  ------------
void CosmicRateAnalyzer::endJob() {
  std::cout << "Total Events: " << nTotalEvents << std::endl;
  std::cout << "TotalTracks: " << nTotalTracks << std::endl;
}

// ------------ method called when starting to processes a run  ------------
void CosmicRateAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&) {
  lastruntime = 0.0;
  lastrunnum = 0.0;
  ntrk_runnum = 0.0;
  events = 0.0;
  // a) for Track rate
  track_BPIX = 0.0;
  track_FPIX = 0.0;
  track_PIXEL = 0.0;
  track_TEC = 0.0;
  track_TECM = 0.0;
  track_TECP = 0.0;
  track_TOB = 0.0;
  track_TIB = 0.0;
  track_TID = 0.0;
  track_TIDM = 0.0;
  track_TIDP = 0.0;

  // b) for PIXEL hit rate
  hit_Total = 0.0;
  hit_PIX = 0.0;
  hit_BPIX = 0.0;
  hit_BPIX_layer1 = 0.0;
  hit_BPIX_layer2 = 0.0;
  hit_BPIX_layer3 = 0.0;
  hit_BPIX_layer4 = 0.0;
  hit_FPIX = 0.0;
  hit_FPIX_disk1 = 0.0;
  hit_FPIX_disk2 = 0.0;
  hit_FPIX_disk3 = 0.0;
  hit_FPIX_disk1_plus = 0.0;
  hit_FPIX_disk1_minus = 0.0;
  hit_FPIX_disk2_plus = 0.0;
  hit_FPIX_disk2_minus = 0.0;
  hit_FPIX_disk3_plus = 0.0;
  hit_FPIX_disk3_minus = 0.0;
}

// ------------ method called when ending the processing of a run  ------------

void CosmicRateAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) {
  number_of_tracks = ntrk_runnum;
  run_time = lastruntime;
  runnum = lastrunnum;
  // a) for Track Rate
  number_of_tracks_PIX = track_PIXEL;
  number_of_tracks_FPIX = track_FPIX;
  number_of_tracks_BPIX = track_BPIX;
  number_of_tracks_TEC = track_TEC;
  number_of_tracks_TECM = track_TECM;
  number_of_tracks_TECP = track_TECP;
  number_of_tracks_TOB = track_TOB;
  number_of_tracks_TIB = track_TIB;
  number_of_tracks_TID = track_TID;
  number_of_tracks_TIDM = track_TIDM;
  number_of_tracks_TIDP = track_TIDP;
  number_of_events = events;
  // b) for PIXEL Hit Rate
  number_of_hits_Total = hit_Total;
  number_of_hits_PIX = hit_PIX;
  number_of_hits_BPIX = hit_BPIX;
  number_of_hits_BPIX_layer1 = hit_BPIX_layer1;
  number_of_hits_BPIX_layer2 = hit_BPIX_layer2;
  number_of_hits_BPIX_layer3 = hit_BPIX_layer3;
  number_of_hits_BPIX_layer4 = hit_BPIX_layer4;
  number_of_hits_FPIX = hit_FPIX;
  number_of_hits_FPIX_disk1 = hit_FPIX_disk1;
  number_of_hits_FPIX_disk2 = hit_FPIX_disk2;
  number_of_hits_FPIX_disk3 = hit_FPIX_disk3;
  number_of_hits_FPIX_disk1_plus = hit_FPIX_disk1_plus;
  number_of_hits_FPIX_disk1_minus = hit_FPIX_disk1_minus;
  number_of_hits_FPIX_disk2_plus = hit_FPIX_disk2_plus;
  number_of_hits_FPIX_disk2_minus = hit_FPIX_disk2_minus;
  number_of_hits_FPIX_disk3_plus = hit_FPIX_disk3_plus;
  number_of_hits_FPIX_disk3_minus = hit_FPIX_disk3_minus;

  treeRun->Fill();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CosmicRateAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Create tuple with all variables required to calculate cosmic event and track rates.");
  desc.add<edm::InputTag>("tracksInputTag", edm::InputTag("ALCARECOTkAlCosmicsCTF0T"));
  desc.add<edm::InputTag>("muonsInputTag", edm::InputTag("muons1Leg"));
  descriptions.add("cosmicRateAnalyzer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CosmicRateAnalyzer);
