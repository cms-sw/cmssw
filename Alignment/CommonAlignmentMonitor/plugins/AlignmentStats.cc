// system includes
#include <fstream>
#include <string>
#include <vector>
#include <map>

// user includes
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"
#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Utilities/General/interface/ClassName.h"

// ROOT includes
#include "TFile.h"
#include "TTree.h"

namespace aliStats {
  struct GeoInfo {
  public:
    void printAll() const {
      edm::LogInfo("GeoInfo") << "DetId: " << id_ << " subdet: " << subdet_ << " layer:" << layer_
                              << " (pox,posy,posz) = (" << posX_ << "," << posY_ << "," << posZ_ << ")"
                              << " posEta: " << posEta_ << " posPhi: " << posPhi_ << " posR: " << posR_
                              << " is2D:" << is2D_ << " isStereo:" << isStereo_;
    }
    unsigned int id_;
    float posX_;
    float posY_;
    float posZ_;
    float posEta_;
    float posPhi_;
    float posR_;
    int subdet_;
    unsigned int layer_;
    bool is2D_;
    bool isStereo_;
  };
}  // namespace aliStats

class AlignmentStats : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  AlignmentStats(const edm::ParameterSet &iConfig);
  ~AlignmentStats() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginRun(edm::Run const &, edm::EventSetup const &) override{};
  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;
  void endRun(edm::Run const &iRun, edm::EventSetup const &iSetup) override;
  void beginJob() override;
  void endJob() override;

private:
  std::vector<aliStats::GeoInfo> geomInfoList_;

  // esToken
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> esTokenTTopoER_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> esTokenTkGeoER_;

  //////inputs from config file
  const edm::InputTag src_;
  const edm::InputTag overlapAM_;
  const bool keepTrackStats_;
  const bool keepHitPopulation_;
  const std::string statsTreeName_;
  const std::string hitsTreeName_;
  const uint32_t prescale_;

  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const edm::EDGetTokenT<AliClusterValueMap> mapToken_;

  //////
  uint32_t tmpPresc_;

  //Track stats
  TFile *treefile_;
  TTree *outtree_;
  static const int MAXTRKS_ = 200;
  int run_, event_;
  unsigned int ntracks;
  float P[MAXTRKS_], Pt[MAXTRKS_], Eta[MAXTRKS_], Phi[MAXTRKS_], Chi2n[MAXTRKS_];
  int Nhits[MAXTRKS_][7];  //0=total, 1-6=Subdets

  //Hit Population
  TFile *hitsfile_;
  TTree *hitstree_;
  unsigned int id_, nhits_, noverlaps_;
  float posX_, posY_, posZ_;
  float posEta_, posPhi_, posR_;
  int subdet_;
  unsigned int layer_;
  bool is2D_, isStereo_;

  typedef std::map<uint32_t, uint32_t> DetHitMap;
  DetHitMap hitmap_;
  DetHitMap overlapmap_;

  std::unique_ptr<TrackerTopology> trackerTopology_;
};

using namespace std;

AlignmentStats::AlignmentStats(const edm::ParameterSet &iConfig)
    : esTokenTTopoER_(esConsumes<edm::Transition::EndRun>()),
      esTokenTkGeoER_(esConsumes<edm::Transition::EndRun>()),
      src_(iConfig.getParameter<edm::InputTag>("src")),
      overlapAM_(iConfig.getParameter<edm::InputTag>("OverlapAssoMap")),
      keepTrackStats_(iConfig.getParameter<bool>("keepTrackStats")),
      keepHitPopulation_(iConfig.getParameter<bool>("keepHitStats")),
      statsTreeName_(iConfig.getParameter<string>("TrkStatsFileName")),
      hitsTreeName_(iConfig.getParameter<string>("HitStatsFileName")),
      prescale_(iConfig.getParameter<uint32_t>("TrkStatsPrescale")),
      trackToken_(consumes<reco::TrackCollection>(src_)),
      mapToken_(consumes<AliClusterValueMap>(overlapAM_)) {
  //sanity checks

  //init
  outtree_ = nullptr;

}  //end constructor

void AlignmentStats::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("OverlapAssoMap", edm::InputTag("OverlapAssoMap"));
  desc.add<bool>("keepTrackStats", false);
  desc.add<bool>("keepHitStats", false);
  desc.add<std::string>("TrkStatsFileName", "tracks_statistics.root");
  desc.add<std::string>("HitStatsFileName", "HitMaps.root");
  desc.add<unsigned int>("TrkStatsPrescale", 1);
  descriptions.add("AlignmentStats", desc);
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

  // create tree with hit maps (hitstree)
  // book hits stats tree
  hitsfile_ = new TFile(hitsTreeName_.c_str(), "RECREATE");
  hitsfile_->cd();
  hitstree_ = new TTree("AlignmentHitMap", "Maps of Hits used for Alignment");
  hitstree_->Branch("DetId", &id_, "DetId/i");
  hitstree_->Branch("Nhits", &nhits_, "Nhits/i");
  hitstree_->Branch("Noverlaps", &noverlaps_, "Noverlaps/i");
  hitstree_->Branch("SubDet", &subdet_, "SubDet/I");
  hitstree_->Branch("Layer", &layer_, "Layer/i");
  hitstree_->Branch("is2D", &is2D_, "is2D/B");
  hitstree_->Branch("isStereo", &isStereo_, "isStereo/B");
  hitstree_->Branch("posX", &posX_, "posX/F");
  hitstree_->Branch("posY", &posY_, "posY/F");
  hitstree_->Branch("posZ", &posZ_, "posZ/F");
  hitstree_->Branch("posR", &posR_, "posR/F");
  hitstree_->Branch("posEta", &posEta_, "posEta/F");
  hitstree_->Branch("posPhi", &posPhi_, "posPhi/F");
}  //end beginJob

// ------------ method called once every run before doing the event loop ----------------
void AlignmentStats::endRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  if (!trackerTopology_) {
    trackerTopology_ = std::make_unique<TrackerTopology>(iSetup.getData(esTokenTTopoER_));
    const TrackerGeometry *trackerGeometry_ = &iSetup.getData(esTokenTkGeoER_);
    auto theAliTracker = std::make_unique<AlignableTracker>(trackerGeometry_, trackerTopology_.get());

    hitsfile_->cd();
    for (const auto &detUnit : theAliTracker->deepComponents()) {
      aliStats::GeoInfo detUnitInfo;
      detUnitInfo.id_ = static_cast<uint32_t>(detUnit->id());
      DetId detid(detUnitInfo.id_);
      detUnitInfo.subdet_ = detid.subdetId();

      //take other geometrical infos from the det
      detUnitInfo.posX_ = detUnit->globalPosition().x();
      detUnitInfo.posY_ = detUnit->globalPosition().y();
      detUnitInfo.posZ_ = detUnit->globalPosition().z();

      align::GlobalVector vec(detUnitInfo.posX_, detUnitInfo.posY_, detUnitInfo.posZ_);
      detUnitInfo.posR_ = vec.perp();
      detUnitInfo.posPhi_ = vec.phi();
      detUnitInfo.posEta_ = vec.eta();
      // detUnitInfo.posPhi_ = atan2(posY_,posX_);

      //get layers, petals, etc...
      if (detUnitInfo.subdet_ == PixelSubdetector::PixelBarrel) {  //PXB
        detUnitInfo.layer_ = trackerTopology_->pxbLayer(detUnitInfo.id_);
        detUnitInfo.is2D_ = true;
        detUnitInfo.isStereo_ = false;
      } else if (detUnitInfo.subdet_ == PixelSubdetector::PixelEndcap) {
        detUnitInfo.layer_ = trackerTopology_->pxfDisk(detUnitInfo.id_);
        detUnitInfo.is2D_ = true;
        detUnitInfo.isStereo_ = false;
      } else if (detUnitInfo.subdet_ == SiStripDetId::TIB) {
        detUnitInfo.layer_ = trackerTopology_->tibLayer(detUnitInfo.id_);
        detUnitInfo.is2D_ = trackerTopology_->tibIsDoubleSide(detUnitInfo.id_);
        detUnitInfo.isStereo_ = trackerTopology_->tibIsStereo(detUnitInfo.id_);
      } else if (detUnitInfo.subdet_ == SiStripDetId::TID) {
        detUnitInfo.layer_ = trackerTopology_->tidWheel(detUnitInfo.id_);
        detUnitInfo.is2D_ = trackerTopology_->tidIsDoubleSide(detUnitInfo.id_);
        detUnitInfo.isStereo_ = trackerTopology_->tidIsStereo(detUnitInfo.id_);
      } else if (detUnitInfo.subdet_ == SiStripDetId::TOB) {
        detUnitInfo.layer_ = trackerTopology_->tobLayer(detUnitInfo.id_);
        detUnitInfo.is2D_ = trackerTopology_->tobIsDoubleSide(detUnitInfo.id_);
        detUnitInfo.isStereo_ = trackerTopology_->tobIsStereo(detUnitInfo.id_);
      } else if (detUnitInfo.subdet_ == SiStripDetId::TEC) {
        detUnitInfo.layer_ = trackerTopology_->tecWheel(detUnitInfo.id_);
        detUnitInfo.is2D_ = trackerTopology_->tecIsDoubleSide(detUnitInfo.id_);
        detUnitInfo.isStereo_ = trackerTopology_->tecIsStereo(detUnitInfo.id_);
      } else {
        edm::LogError("AlignmentStats")
            << "Detector not belonging neither to pixels nor to strips! Skipping it. SubDet= " << detUnitInfo.subdet_;
      }

      LogDebug("AlignmentStats") << "id " << detUnitInfo.id_ << " detid.rawId()" << detid.rawId() << " subdet "
                                 << detUnitInfo.subdet_;

      // push back in the list
      geomInfoList_.push_back(detUnitInfo);
    }  // end loop over detunits

    int ndetunits = geomInfoList_.size();
    edm::LogInfo("AlignmentStats") << __PRETTY_FUNCTION__
                                   << " Number of DetUnits in the AlignableTracker: " << ndetunits;
  }
}

void AlignmentStats::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  //take trajectories and tracks to loop on
  // edm::Handle<TrajTrackAssociationCollection> TrackAssoMap;
  const edm::Handle<reco::TrackCollection> &Tracks = iEvent.getHandle(trackToken_);

  //take overlap HitAssomap
  const edm::Handle<AliClusterValueMap> &hMap = iEvent.getHandle(mapToken_);
  const AliClusterValueMap &OverlapMap = *hMap;

  // Initialise
  run_ = 1;
  event_ = 1;
  ntracks = 0;
  run_ = iEvent.id().run();
  event_ = iEvent.id().event();
  ntracks = Tracks->size();
  if (ntracks > 1)
    edm::LogVerbatim("AlignmenStats") << "~~~~~~~~~~~~\n For this event processing " << ntracks << " tracks";

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
  for (const auto &ittrk : *Tracks) {
    Eta[trk_cnt] = ittrk.eta();
    Phi[trk_cnt] = ittrk.phi();
    Chi2n[trk_cnt] = ittrk.normalizedChi2();
    P[trk_cnt] = ittrk.p();
    Pt[trk_cnt] = ittrk.pt();
    Nhits[trk_cnt][0] = ittrk.numberOfValidHits();

    if (ntracks > 1)
      edm::LogVerbatim("AlignmenStats") << "Track #" << trk_cnt + 1 << " params:    Eta=" << Eta[trk_cnt]
                                        << "  Phi=" << Phi[trk_cnt] << "  P=" << P[trk_cnt]
                                        << "   Nhits=" << Nhits[trk_cnt][0];

    //loop on tracking rechits
    //edm::LogVerbatim("AlignmenStats") << "   loop on hits of track #" << (itt - tracks->begin());
    for (auto const &hit : ittrk.recHits()) {
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
            //edm::LogVerbatim("AlignmenStats")<<"Taken the Strip Cluster with ProdId "<<stripclust.id() <<"; the Value in the map is "<<inval<<"  (DetId is "<<hit->geographicalId().rawId()<<")";
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
          //edm::LogVerbatim("AlignmenStats")<<"Taken the Pixel Cluster with ProdId "<<pixclust.id() <<"; the Value in the map is "<<inval<<"  (DetId is "<<hit->geographicalId().rawId()<<")";
        } else {
          edm::LogError("AlignmentStats")
              << "ERROR in <AlignmentStats::analyze>: Dynamic cast of Pixel RecHit failed!   TypeId of the RecHit: "
              << className(*hit);
        }
      }  //end else hit is in Pixel

      bool isOverlapHit(inval.isOverlap());

      if (isOverlapHit) {
        edm::LogVerbatim("AlignmenStats") << "This hit is an overlap !";
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
      if (ntracks > 1)
        edm::LogVerbatim("AlignmenStats") << "Hit in SubDet=" << subdethit;
      Nhits[trk_cnt][subdethit] = Nhits[trk_cnt][subdethit] + 1;
    }  //end loop on trackingrechits
    trk_cnt++;

  }  //end loop on tracks

  //edm::LogVerbatim("AlignmenStats")<<"Total number of pixel hits is " << npxbhits;

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
  edm::LogInfo("AlignmentStats") << "Writing out the TrackStatistics in " << gDirectory->GetPath();
  outtree_->Write();
  delete outtree_;

  int ndetunits = geomInfoList_.size();
  edm::LogInfo("AlignmentStats") << __PRETTY_FUNCTION__ << "Number of DetUnits in the AlignableTracker: " << ndetunits
                                 << std::endl;

  hitsfile_->cd();
  // save the information on hits and overlaps
  for (const auto &detUnitInfo : geomInfoList_) {
    detUnitInfo.printAll();

    // copy the values from the struct to the TTree
    id_ = detUnitInfo.id_;
    posX_ = detUnitInfo.posX_;
    posY_ = detUnitInfo.posY_;
    posZ_ = detUnitInfo.posZ_;
    posEta_ = detUnitInfo.posEta_;
    posPhi_ = detUnitInfo.posPhi_;
    posR_ = detUnitInfo.posR_;
    subdet_ = detUnitInfo.subdet_;
    layer_ = detUnitInfo.layer_;
    is2D_ = detUnitInfo.is2D_;
    isStereo_ = detUnitInfo.isStereo_;

    if (hitmap_.find(id_) != hitmap_.end()) {
      nhits_ = hitmap_[id_];
    }
    //if not, save nhits=0
    else {
      nhits_ = 0;
      hitmap_.insert(pair<uint32_t, uint32_t>(id_, 0));
    }

    if (overlapmap_.find(id_) != overlapmap_.end()) {
      noverlaps_ = overlapmap_[id_];
    }
    //if not, save nhits=0
    else {
      noverlaps_ = 0;
      overlapmap_.insert(pair<uint32_t, uint32_t>(id_, 0));
    }
    //write in the hitstree
    hitstree_->Fill();
  }  //end loop over detunits

  //save hitstree
  hitstree_->Write();
  delete hitstree_;
  hitmap_.clear();
  overlapmap_.clear();
  delete hitsfile_;
}
// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlignmentStats);
