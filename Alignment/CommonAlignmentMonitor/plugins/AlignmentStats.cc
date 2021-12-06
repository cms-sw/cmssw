#include "Alignment/CommonAlignmentMonitor/plugins/AlignmentStats.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"

#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "Utilities/General/interface/ClassName.h"

using namespace std;

AlignmentStats::AlignmentStats(const edm::ParameterSet &iConfig)
    : esTokenTTopo_(esConsumes()),
      esTokenTkGeo_(esConsumes()),
      src_(iConfig.getParameter<edm::InputTag>("src")),
      overlapAM_(iConfig.getParameter<edm::InputTag>("OverlapAssoMap")),
      keepTrackStats_(iConfig.getParameter<bool>("keepTrackStats")),
      keepHitPopulation_(iConfig.getParameter<bool>("keepHitStats")),
      statsTreeName_(iConfig.getParameter<string>("TrkStatsFileName")),
      hitsTreeName_(iConfig.getParameter<string>("HitStatsFileName")),
      prescale_(iConfig.getParameter<uint32_t>("TrkStatsPrescale")) {
  //sanity checks

  //init
  outtree_ = nullptr;

}  //end constructor

AlignmentStats::~AlignmentStats() {
  //
}

void AlignmentStats::beginJob() {  // const edm::EventSetup &iSetup

  //book track stats tree
  treefile_ = new TFile(statsTreeName_.c_str(), "RECREATE");
  treefile_->cd();
  outtree_ = new TTree("AlignmentTrackStats", "Statistics of Tracks used for Alignment");
  // int nHitsinPXB[MAXTRKS_], nHitsinPXE[MAXTRKS_], nHitsinTEC[MAXTRKS_], nHitsinTIB[MAXTRKS_],nHitsinTOB[MAXTRKS_],nHitsinTID[MAXTRKS_];

  outtree_->Branch("Ntracks", &ntracks, "Ntracks/i");
  outtree_->Branch("Run_", &run_, "Run_Nr/I");
  outtree_->Branch("Event", &event_, "EventNr/I");
  outtree_->Branch("Eta", &Eta, "Eta[Ntracks]/F");
  outtree_->Branch("Phi", &Phi, "Phi[Ntracks]/F");
  outtree_->Branch("P", &P, "P[Ntracks]/F");
  outtree_->Branch("Pt", &Pt, "Pt[Ntracks]/F");
  outtree_->Branch("Chi2n", &Chi2n, "Chi2n[Ntracks]/F");
  outtree_->Branch("Nhits", &Nhits, "Nhits[Ntracks][7]/I");
  /*
    outtree_->Branch("NhitsPXB"       , ,);
    outtree_->Branch("NhitsPXE"       , ,);
    outtree_->Branch("NhitsTIB"       , ,);
    outtree_->Branch("NhitsTID"       , ,);
    outtree_->Branch("NhitsTOB"       , ,);
    outtree_->Branch("NhitsTOB"       , ,);
  */

  tmpPresc_ = prescale_;
}  //end beginJob

void AlignmentStats::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  //load list of detunits needed then in endJob
  if (!trackerGeometry_) {
    trackerGeometry_ = std::make_unique<TrackerGeometry>(iSetup.getData(esTokenTkGeo_));
  }

  if (!trackerTopology_) {
    trackerTopology_ = std::make_unique<TrackerTopology>(iSetup.getData(esTokenTTopo_));
  }

  //take trajectories and tracks to loop on
  // edm::Handle<TrajTrackAssociationCollection> TrackAssoMap;
  edm::Handle<reco::TrackCollection> Tracks;
  iEvent.getByLabel(src_, Tracks);

  //take overlap HitAssomap
  edm::Handle<AliClusterValueMap> hMap;
  iEvent.getByLabel(overlapAM_, hMap);
  const AliClusterValueMap &OverlapMap = *hMap;

  // Initialise
  run_ = 1;
  event_ = 1;
  ntracks = 0;
  run_ = iEvent.id().run();
  event_ = iEvent.id().event();
  ntracks = Tracks->size();
  //  if(ntracks>1) std::cout<<"~~~~~~~~~~~~\n For this event processing "<<ntracks<<" tracks"<<std::endl;

  unsigned int trk_cnt = 0;

  for (int j = 0; j < MAXTRKS_; j++) {
    Eta[j] = -9999.0;
    Phi[j] = -8888.0;
    P[j] = -7777.0;
    Pt[j] = -6666.0;
    Chi2n[j] = -2222.0;
    for (int k = 0; k < 7; k++) {
      Nhits[j][k] = 0;
    }
  }

  // int npxbhits=0;

  //loop on tracks
  for (std::vector<reco::Track>::const_iterator ittrk = Tracks->begin(), edtrk = Tracks->end(); ittrk != edtrk;
       ++ittrk) {
    Eta[trk_cnt] = ittrk->eta();
    Phi[trk_cnt] = ittrk->phi();
    Chi2n[trk_cnt] = ittrk->normalizedChi2();
    P[trk_cnt] = ittrk->p();
    Pt[trk_cnt] = ittrk->pt();
    Nhits[trk_cnt][0] = ittrk->numberOfValidHits();

    //   if(ntracks>1)std::cout<<"Track #"<<trk_cnt+1<<" params:    Eta="<< Eta[trk_cnt]<<"  Phi="<< Phi[trk_cnt]<<"  P="<<P[trk_cnt]<<"   Nhits="<<Nhits[trk_cnt][0]<<std::endl;

    int nhit = 0;
    //loop on tracking rechits
    //std::cout << "   loop on hits of track #" << (itt - tracks->begin()) << std::endl;
    for (auto const &hit : ittrk->recHits()) {
      if (!hit->isValid())
        continue;
      DetId detid = hit->geographicalId();
      int subDet = detid.subdetId();
      uint32_t rawId = hit->geographicalId().rawId();

      //  if(subDet==1)npxbhits++;

      //look if you find this detid in the map
      DetHitMap::iterator mapiter;
      mapiter = hitmap_.find(rawId);
      if (mapiter != hitmap_.end()) {  //present, increase its value by one
        //	hitmap_[rawId]=hitmap_[rawId]+1;
        ++(hitmap_[rawId]);
      } else {  //not present, let's add this key to the map with value=1
        hitmap_.insert(pair<uint32_t, uint32_t>(rawId, 1));
      }

      AlignmentClusterFlag inval;

      bool hitInPixel = (subDet == PixelSubdetector::PixelBarrel) || (subDet == PixelSubdetector::PixelEndcap);
      bool hitInStrip = (subDet == SiStripDetId::TIB) || (subDet == SiStripDetId::TID) ||
                        (subDet == SiStripDetId::TOB) || (subDet == SiStripDetId::TEC);

      if (!(hitInPixel || hitInStrip)) {
        //skip only this hit, don't stop everything throwing an exception
        edm::LogError("AlignmentStats") << "Hit not belonging neither to pixels nor to strips! Skipping it. SubDet= "
                                        << subDet;
        continue;
      }

      //check also if the hit is an overlap. If yes fill a dedicated hitmap
      if (hitInStrip) {
        const std::type_info &type = typeid(*hit);

        if (type == typeid(SiStripRecHit1D)) {
          //Notice the difference respect to when one loops on Trajectories: the recHit is a TrackingRecHit and not a TransientTrackingRecHit
          const SiStripRecHit1D *striphit = dynamic_cast<const SiStripRecHit1D *>(hit);
          if (striphit != nullptr) {
            SiStripRecHit1D::ClusterRef stripclust(striphit->cluster());
            inval = OverlapMap[stripclust];
          } else {
            //  edm::LogError("AlignmentStats")<<"ERROR in <AlignmentStats::analyze>: Dynamic cast of Strip RecHit1D failed!   TypeId of the RecHit: "<<className(*hit);
            throw cms::Exception("NullPointerError")
                << "ERROR in <AlignmentStats::analyze>: Dynamic cast of Strip RecHit1D failed!   TypeId of the RecHit: "
                << className(*hit);
          }
        }  //end if sistriprechit1D
        if (type == typeid(SiStripRecHit2D)) {
          const SiStripRecHit2D *striphit = dynamic_cast<const SiStripRecHit2D *>(hit);
          if (striphit != nullptr) {
            SiStripRecHit2D::ClusterRef stripclust(striphit->cluster());
            inval = OverlapMap[stripclust];
            //cout<<"Taken the Strip Cluster with ProdId "<<stripclust.id() <<"; the Value in the map is "<<inval<<"  (DetId is "<<hit->geographicalId().rawId()<<")"<<endl;
          } else {
            throw cms::Exception("NullPointerError")
                << "ERROR in <AlignmentStats::analyze>: Dynamic cast of Strip RecHit2D failed!   TypeId of the RecHit: "
                << className(*hit);
            //	  edm::LogError("AlignmentStats")<<"ERROR in <AlignmentStats::analyze>: Dynamic cast of Strip RecHit2D failed!   TypeId of the RecHit: "<<className(*hit);
          }
        }  //end if sistriprechit2D

      }  //end if hit in Strips
      else {
        const SiPixelRecHit *pixelhit = dynamic_cast<const SiPixelRecHit *>(hit);
        if (pixelhit != nullptr) {
          SiPixelClusterRefNew pixclust(pixelhit->cluster());
          inval = OverlapMap[pixclust];
          //cout<<"Taken the Pixel Cluster with ProdId "<<pixclust.id() <<"; the Value in the map is "<<inval<<"  (DetId is "<<hit->geographicalId().rawId()<<")"<<endl;
        } else {
          edm::LogError("AlignmentStats")
              << "ERROR in <AlignmentStats::analyze>: Dynamic cast of Pixel RecHit failed!   TypeId of the RecHit: "
              << className(*hit);
        }
      }  //end else hit is in Pixel

      bool isOverlapHit(inval.isOverlap());

      if (isOverlapHit) {
        //cout<<"This hit is an overlap !"<<endl;
        DetHitMap::iterator overlapiter;
        overlapiter = overlapmap_.find(rawId);

        if (overlapiter !=
            overlapmap_.end()) {  //the det already collected at least an overlap, increase its value by one
          overlapmap_[rawId] = overlapmap_[rawId] + 1;
        } else {  //first overlap on det unit, let's add it to the map
          overlapmap_.insert(pair<uint32_t, uint32_t>(rawId, 1));
        }
      }  //end if the hit is an overlap

      int subdethit = static_cast<int>(hit->geographicalId().subdetId());
      // if(ntracks>1)std::cout<<"Hit in SubDet="<<subdethit<<std::endl;
      Nhits[trk_cnt][subdethit] = Nhits[trk_cnt][subdethit] + 1;
      nhit++;
    }  //end loop on trackingrechits
    trk_cnt++;

  }  //end loop on tracks

  //  cout<<"Total number of pixel hits is "<<npxbhits<<endl;

  tmpPresc_--;
  if (tmpPresc_ < 1) {
    outtree_->Fill();
    tmpPresc_ = prescale_;
  }
  if (trk_cnt != ntracks)
    edm::LogError("AlignmentStats") << "\nERROR! trk_cnt=" << trk_cnt << "   ntracks=" << ntracks;

  return;
}

void AlignmentStats::endJob() {
  treefile_->cd();
  edm::LogInfo("AlignmentStats") << "Writing out the TrackStatistics in " << gDirectory->GetPath() << std::endl;
  outtree_->Write();
  delete outtree_;

  //create tree with hit maps (hitstree)
  //book track stats tree
  TFile *hitsfile = new TFile(hitsTreeName_.c_str(), "RECREATE");
  hitsfile->cd();
  TTree *hitstree = new TTree("AlignmentHitMap", "Maps of Hits used for Alignment");

  unsigned int id = 0, nhits = 0, noverlaps = 0;
  float posX(-99999.0), posY(-77777.0), posZ(-88888.0);
  float posEta(-6666.0), posPhi(-5555.0), posR(-4444.0);
  int subdet = 0;
  unsigned int layer = 0;
  bool is2D = false, isStereo = false;
  hitstree->Branch("DetId", &id, "DetId/i");
  hitstree->Branch("Nhits", &nhits, "Nhits/i");
  hitstree->Branch("Noverlaps", &noverlaps, "Noverlaps/i");
  hitstree->Branch("SubDet", &subdet, "SubDet/I");
  hitstree->Branch("Layer", &layer, "Layer/i");
  hitstree->Branch("is2D", &is2D, "is2D/B");
  hitstree->Branch("isStereo", &isStereo, "isStereo/B");
  hitstree->Branch("posX", &posX, "posX/F");
  hitstree->Branch("posY", &posY, "posY/F");
  hitstree->Branch("posZ", &posZ, "posZ/F");
  hitstree->Branch("posR", &posR, "posR/F");
  hitstree->Branch("posEta", &posEta, "posEta/F");
  hitstree->Branch("posPhi", &posPhi, "posPhi/F");

  /*
  TTree *overlapstree=new TTree("OverlapHitMap","Maps of Overlaps used for Alignment");
  hitstree->Branch("DetId",   &id ,     "DetId/i");
  hitstree->Branch("NOverlaps",   &nhits ,  "Nhits/i");
  hitstree->Branch("SubDet",  &subdet,  "SubDet/I");
  hitstree->Branch("Layer",   &layer,   "Layer/i");
  hitstree->Branch("is2D" ,   &is2D,    "is2D/B");
  hitstree->Branch("isStereo",&isStereo,"isStereo/B");
  hitstree->Branch("posX",    &posX,    "posX/F");
  hitstree->Branch("posY",    &posY,    "posY/F");
  hitstree->Branch("posZ",    &posZ,    "posZ/F");
  hitstree->Branch("posR",    &posR,    "posR/F");
  hitstree->Branch("posEta",  &posEta,  "posEta/F");
  hitstree->Branch("posPhi",  &posPhi,  "posPhi/F");
  */

  std::unique_ptr<AlignableTracker> theAliTracker =
      std::make_unique<AlignableTracker>(trackerGeometry_.get(), trackerTopology_.get());
  const auto &Detunitslist = theAliTracker->deepComponents();
  int ndetunits = Detunitslist.size();
  edm::LogInfo("AlignmentStats") << "Number of DetUnits in the AlignableTracker: " << ndetunits << std::endl;

  for (int det_cnt = 0; det_cnt < ndetunits; ++det_cnt) {
    //re-initialize for safety
    id = 0;
    nhits = 0;
    noverlaps = 0;
    posX = -99999.0;
    posY = -77777.0;
    posZ = -88888.0;
    posEta = -6666.0;
    posPhi = -5555.0;
    posR = -4444.0;
    subdet = 0;
    layer = 0;
    is2D = false;
    isStereo = false;

    //if detunit in vector is found also in the map, look for how many hits were collected
    //and save in the tree this number
    id = static_cast<uint32_t>(Detunitslist[det_cnt]->id());
    if (hitmap_.find(id) != hitmap_.end()) {
      nhits = hitmap_[id];
    }
    //if not, save nhits=0
    else {
      nhits = 0;
      hitmap_.insert(pair<uint32_t, uint32_t>(id, 0));
    }

    if (overlapmap_.find(id) != overlapmap_.end()) {
      noverlaps = overlapmap_[id];
    }
    //if not, save nhits=0
    else {
      noverlaps = 0;
      overlapmap_.insert(pair<uint32_t, uint32_t>(id, 0));
    }

    //take other geometrical infos from the det
    posX = Detunitslist[det_cnt]->globalPosition().x();
    posY = Detunitslist[det_cnt]->globalPosition().y();
    posZ = Detunitslist[det_cnt]->globalPosition().z();

    align::GlobalVector vec(posX, posY, posZ);
    posR = vec.perp();
    posPhi = vec.phi();
    posEta = vec.eta();
    //   posPhi = atan2(posY,posX);

    DetId detid(id);
    subdet = detid.subdetId();

    //get layers, petals, etc...
    if (subdet == PixelSubdetector::PixelBarrel) {  //PXB

      layer = trackerTopology_->pxbLayer(id);
      is2D = true;
      isStereo = false;
    } else if (subdet == PixelSubdetector::PixelEndcap) {
      layer = trackerTopology_->pxfDisk(id);
      is2D = true;
      isStereo = false;
    } else if (subdet == SiStripDetId::TIB) {
      layer = trackerTopology_->tibLayer(id);
      is2D = trackerTopology_->tibIsDoubleSide(id);
      isStereo = trackerTopology_->tibIsStereo(id);
    } else if (subdet == SiStripDetId::TID) {
      layer = trackerTopology_->tidWheel(id);
      is2D = trackerTopology_->tidIsDoubleSide(id);
      isStereo = trackerTopology_->tidIsStereo(id);
    } else if (subdet == SiStripDetId::TOB) {
      layer = trackerTopology_->tobLayer(id);
      is2D = trackerTopology_->tobIsDoubleSide(id);
      isStereo = trackerTopology_->tobIsStereo(id);
    } else if (subdet == SiStripDetId::TEC) {
      layer = trackerTopology_->tecWheel(id);
      is2D = trackerTopology_->tecIsDoubleSide(id);
      isStereo = trackerTopology_->tecIsStereo(id);
    } else {
      edm::LogError("AlignmentStats") << "Detector not belonging neither to pixels nor to strips! Skipping it. SubDet= "
                                      << subdet;
    }

    //write in the hitstree
    hitstree->Fill();
  }  //end loop over detunits

  //save hitstree
  hitstree->Write();
  delete hitstree;
  //delete Detunitslist;
  hitmap_.clear();
  overlapmap_.clear();
  delete hitsfile;
}
// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlignmentStats);
