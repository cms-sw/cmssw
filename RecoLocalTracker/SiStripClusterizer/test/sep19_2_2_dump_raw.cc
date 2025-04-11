/*
 * Dump the offline (RECO) clusters' hits info
 *    - offlineClusterTree: cluster collection that is produced at the RECO level
 *      - The cluster collection is the output of SiStripClusterizer module, with the default value being siStripClusters
 */

// system includes
#include <memory>
#include <iostream>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
//ROOT inclusion
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TMath.h"
#include "TList.h"
#include "TString.h"
#include "cluster_property.h"

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;
//
// class decleration
//

class sep19_2_2_dump_raw : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit sep19_2_2_dump_raw(const edm::ParameterSet&);
  ~sep19_2_2_dump_raw() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::InputTag inputTagClusters;

  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  // Event Data
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> clusterToken;

  // Event Setup Data
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;

  TTree* offlineClusterTree;
  edm::Service<TFileService> fs;

  edm::EventNumber_t eventN;
  int runN;
  int lumi;

  // for offlineClusterTree
  uint32_t    detId;
  uint16_t    firstStrip;
  uint16_t    endStrip;
  float       barycenter;
  uint16_t    size;
  int         charge;
  bool low_pt_trk_cluster;
  bool high_pt_trk_cluster;
  int  trk_algo;

  const static int nMax = 800000;
  float       hitX[nMax];
  float       hitY[nMax];
  uint16_t    channel[nMax];
  uint16_t    adc[nMax];
};

sep19_2_2_dump_raw::sep19_2_2_dump_raw(const edm::ParameterSet& conf) {
  inputTagClusters 	 = conf.getParameter<edm::InputTag>("siStripClustersTag");
  clusterToken 		 = consumes<edmNew::DetSetVector<SiStripCluster>>(inputTagClusters);
  tracksToken_           = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("tracks"));

  tkGeomToken_ = esConsumes();

  usesResource("TFileService");

  offlineClusterTree = fs->make<TTree>("offlineClusterTree", "offlineClusterTree");
  offlineClusterTree->Branch("event", &eventN, "event/i");
  offlineClusterTree->Branch("run",   &runN, "run/I");
  offlineClusterTree->Branch("lumi",  &lumi, "lumi/I");

  offlineClusterTree->Branch("detId", &detId, "detId/i");
  offlineClusterTree->Branch("firstStrip", &firstStrip, "firstStrip/s");
  offlineClusterTree->Branch("endStrip", &endStrip, "endStrip/s");
  offlineClusterTree->Branch("barycenter", &barycenter, "barycenter/F");
  offlineClusterTree->Branch("size", &size, "size/s");
  offlineClusterTree->Branch("charge", &charge, "charge/I");
  offlineClusterTree->Branch("low_pt_trk_cluster", &low_pt_trk_cluster, "low_pt_trk_cluster/b");
  offlineClusterTree->Branch("high_pt_trk_cluster", &high_pt_trk_cluster, "high_pt_trk_cluster/b");
  offlineClusterTree->Branch("trk_algo", &trk_algo, "trk_algo/I"); 

  offlineClusterTree->Branch("x", hitX, "x[size]/F");
  offlineClusterTree->Branch("y", hitY, "y[size]/F");
  offlineClusterTree->Branch("channel", channel, "channel[size]/s");
  offlineClusterTree->Branch("adc", adc, "adc[size]/s");


}

sep19_2_2_dump_raw::~sep19_2_2_dump_raw() = default;

void sep19_2_2_dump_raw::analyze(const edm::Event& event, const edm::EventSetup& es) {
  edm::Handle<edmNew::DetSetVector<SiStripCluster>> clusterCollection 		= event.getHandle(clusterToken);
  const auto& tracksHandle = event.getHandle(tracksToken_);

  using namespace edm;

  if (!tracksHandle.isValid()) {
    edm::LogError("flatNtuple_producer") << "No valid track collection found";
    return;
  }
  const reco::TrackCollection& tracks = *tracksHandle;
  std::map<uint32_t, std::vector<cluster_property>> matched_cluster;

  for(unsigned int i=0; i<tracks.size(); i++) {
    
     auto trk = tracks.at(i);
     for (auto ih = trk.recHitsBegin(); ih != trk.recHitsEnd(); ih++) {
         const SiStripCluster* strip=NULL;
         const TrackingRecHit& hit = **ih;
         const DetId detId((hit).geographicalId());
         if (detId.det() == DetId::Tracker) {
           if (detId.subdetId() == kBPIX || detId.subdetId() == kFPIX) continue;  // pixel is always 2D
           else {        // should be SiStrip now
               if (dynamic_cast<const SiStripRecHit1D *>(&hit)) {
                   strip = dynamic_cast<const SiStripRecHit1D *>(&hit)->cluster().get();
               }
               else if ( dynamic_cast<const SiStripRecHit2D *>(&hit)) {
                 //std::cout << "found SiStripRecHit2D " << std::endl;
                 strip = dynamic_cast<const SiStripRecHit2D *>(&hit)->cluster().get();
               }
               else if (dynamic_cast<const SiStripMatchedRecHit2D *>(&hit)) {
                  //std::cout << "found SiStripMatchedRecHit2D " << std::endl;
                  strip = &(dynamic_cast<const SiStripMatchedRecHit2D *>(&hit))->monoCluster();
               }
           }
         }
         if(strip) {
            bool low_pt_trk = trk.pt() < 0.75;
            matched_cluster[detId].emplace_back(
                 low_pt_trk, !low_pt_trk, strip->barycenter(),
                 strip->size(), strip->firstStrip(), strip->endStrip(),
                 strip->charge(),
                 trk.algo()
            );
         }
     }
  }
  
  const auto& tkGeom = &es.getData(tkGeomToken_);
  const auto tkDets = tkGeom->dets();
  for (const auto& detSiStripClusters : *clusterCollection) {
    eventN = event.id().event();
    runN   = (int) event.id().run();
    lumi   = (int) event.id().luminosityBlock();
    detId = detSiStripClusters.detId();
    for (const auto& stripCluster : detSiStripClusters) {
      firstStrip  = stripCluster.firstStrip();
      endStrip    = stripCluster.endStrip();
      barycenter  = stripCluster.barycenter();
      size        = stripCluster.size();
      charge      = stripCluster.charge();
      const auto& _detId = detId; // for the capture clause in the lambda function
      auto det = std::find_if(tkDets.begin(), tkDets.end(), [_detId](auto& elem) -> bool {
        return (elem->geographicalId().rawId() == _detId);
      });
      const StripTopology& p = dynamic_cast<const StripGeomDetUnit*>(*det)->specificTopology();
      for (int strip = firstStrip; strip < endStrip+1; ++strip)
      {
        GlobalPoint gp = (tkGeom->idToDet(detId))->surface().toGlobal(p.localPosition((float) strip));

        hitX   [strip - firstStrip] = gp.x();
        hitY   [strip - firstStrip] = gp.y();
        channel[strip - firstStrip] = strip;
        adc    [strip - firstStrip] = stripCluster[strip - firstStrip];
      }
      
      low_pt_trk_cluster = false;
      high_pt_trk_cluster = false;
      trk_algo            = -1;

      if(matched_cluster.find(detId) != matched_cluster.end())
      { 
        for(auto& trk_cluster_property: matched_cluster[detId])
        {  
           if (trk_cluster_property.barycenter == barycenter)
           {
               assert( (size == trk_cluster_property.size)
                      && (firstStrip == trk_cluster_property.firstStrip)
                      && (endStrip == trk_cluster_property.endStrip)
                      && (charge == trk_cluster_property.charge)
               );
               low_pt_trk_cluster = trk_cluster_property.low_pt_trk_cluster;
               high_pt_trk_cluster = trk_cluster_property.high_pt_trk_cluster;
               trk_algo           = trk_cluster_property.trk_algo;
           }
        }
      }
 
      offlineClusterTree->Fill();
    }
  }
}

void sep19_2_2_dump_raw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("siStripClustersTag", edm::InputTag("siStripClusters"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks","","reRECO"));
  descriptions.add("sep19_2_2_dump_raw", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(sep19_2_2_dump_raw);
