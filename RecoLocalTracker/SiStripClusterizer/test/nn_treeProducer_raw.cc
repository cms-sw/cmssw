/*
 * Dump the online (HLT) clusters' hits info
 *    - bkgTree: (approximated) cluster collection that is produced at the HLT level
 *      - The approximated cluster collection is the output of SiStripClusters2ApproxClusters module, with the default value being hltSiStripClusters2ApproxClusters
 *      - If doDumpInputOfSiStripClusters2ApproxClusters,
 *        The input cluster collection of SiStripClusters2ApproxClusters would also be stored out, with the default value being hltSiStripClusterizerForRawPrime
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
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
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
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

#include "assert.h"
//ROOT inclusion
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TMath.h"
#include "TList.h"
#include "TString.h"
#include "TH2F.h"
#include "cluster_property.h"

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;
//
// class decleration
//

class nn_tupleProducer_raw : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit nn_tupleProducer_raw(const edm::ParameterSet&);
  ~nn_tupleProducer_raw() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  bool doDumpInputOfSiStripClusters2ApproxClusters;

  edm::InputTag inputTagApproxClusters;
  edm::InputTag inputTagClusters;

  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<reco::TrackCollection> hlttracksToken_; 

  // Event Data
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> clusterToken;

  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> stripNoiseToken_;
  edm::ESHandle<SiStripNoises> theNoise_;

  // Event Setup Data
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;

  // Strip CPE
  edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord> stripCPEToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
 //edm::typelookup::classTypeInfo<MagneticField const> magFieldToken_;

  TTree* signalTree;
  TTree* bkgTree;
  edm::Service<TFileService> fs;

  edm::EventNumber_t eventN;
  int runN;
  int lumi;

  static constexpr double subclusterWindow_ = .7;
  static constexpr double seedCutMIPs_ = .35;
  static constexpr double seedCutSN_ = 7.;
  static constexpr double subclusterCutMIPs_ = .45;
  static constexpr double subclusterCutSN_ = 12.;

  edm::InputTag beamSpot_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  edm::FileInPath fileInPath_;
  SiStripDetInfo detInfo_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbToken_;
  const TransientTrackBuilder* theTTrackBuilder;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  // for approxCluster
  uint32_t    detId;
  uint16_t    firstStrip;
  uint16_t    endStrip;
  float       barycenter;
  float       vrtx_xy;
  float       vrtx_z;
  UShort_t    falling_barycenter;
  uint16_t    size;
  uint16_t    maxSat;

  int         charge;
  int         max_adc;
  int         max_adc_idx;
  float       max_adc_x;
  float       max_adc_y;
  float       max_adc_z;
  float       dr_min_pixelTrk;
  int         adc_mone;
  int         adc_mtwo;
  int         adc_mthree;
  int         adc_mfour;
  int         adc_pone;
  int         adc_ptwo;
  int         adc_pthree;
  int         adc_pfour;
  bool        low_pt_trk_cluster;
  bool        high_pt_trk_cluster;
  int         trk_algo;
  bool        saturated;
  bool        isTIB;
  bool        isTOB;
  bool        isTID;
  bool        isTEC;
  bool        isStereo;
  bool        isglued;
  bool        isstacked;

  const static int nMax = 8000000;
  float       hitX[nMax];
  float       hitY[nMax];
  float       hitZ[nMax];
  uint16_t    channel[nMax];
  uint16_t    adc[nMax];
  int    adc_zero;
  int    adc_one;
  int    adc_two;
  int    adc_three;
  int    adc_four;
  float       adc_std;
  unsigned int    layer;

  // for reference of approxCluster
  uint16_t    ref_firstStrip;
  uint16_t    ref_endStrip;
  float       ref_barycenter;
  uint16_t    ref_size;
  int         ref_charge;
  int         diff_adc_mone;
  int         diff_adc_mtwo;
  int         diff_adc_mthree;
  int         diff_adc_pone;
  int         diff_adc_ptwo;
  int         diff_adc_pthree;
  float       noise_max_adc;
  float       noise_adc_mone;
  float       noise_adc_mtwo;
  float       noise_adc_mthree;
  float       noise_adc_pone;
  float       noise_adc_ptwo;
  float       noise_adc_pthree;
  int         n_saturated;
  int         n_consecutive_saturated;

  float       ref_hitX[nMax];
  float       trk_pt;
  float       ref_hitY[nMax];
  uint16_t    ref_channel[nMax];
  uint16_t    ref_adc[nMax];

  TH2F* hist_sig;
  TH2F* hist_bkg;
};

nn_tupleProducer_raw::nn_tupleProducer_raw(const edm::ParameterSet& conf):
  magFieldToken_(esConsumes())
  ,propagatorToken_(esConsumes(edm::ESInputTag("", "PropagatorWithMaterialParabolicMf")))
  ,ttbToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder")))
  {
  inputTagClusters       = conf.getParameter<edm::InputTag>("siStripClustersTag");
  vertexToken_ = consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("vertex"));
  clusterToken           = consumes<edmNew::DetSetVector<SiStripCluster>>(inputTagClusters);
  tracksToken_           = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("tracks"));
  hlttracksToken_           = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("hlttracks"));
  stripCPEToken_         = esConsumes<StripClusterParameterEstimator, TkStripCPERecord>(edm::ESInputTag("", "StripCPEfromTrackAngle"));
  tTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  tkGeomToken_ = esConsumes();
  usesResource("TFileService");

  beamSpot_ = conf.getParameter<edm::InputTag>("beamSpot");
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpot_);

  stripNoiseToken_ = esConsumes();

  fileInPath_ = edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile);
  detInfo_ = SiStripDetInfoFileReader::read(fileInPath_.fullPath());
  hist_sig = fs->make<TH2F>("adc_idx_sig","", 50,0,50,260,0,260);
  hist_bkg = fs->make<TH2F>("adc_idx_bkg","", 50,0,50,260,0,260);
  signalTree = fs->make<TTree>("signalTree", "signalTree");
  signalTree->Branch("event", &eventN, "event/i");
  signalTree->Branch("run",   &runN, "run/I");
  signalTree->Branch("lumi",  &lumi, "lumi/I");

  signalTree->Branch("detId", &detId, "detId/i");
  signalTree->Branch("firstStrip", &firstStrip, "firstStrip/s");
  signalTree->Branch("endStrip", &endStrip, "endStrip/s");
  signalTree->Branch("barycenter", &barycenter, "barycenter/F");
  signalTree->Branch("vrtx_xy", &vrtx_xy, "vrtx_xy/F");
  signalTree->Branch("vrtx_z", &vrtx_z, "vrtx_z/F");
  signalTree->Branch("trk_pt", &trk_pt, "trk_pt/F");
  signalTree->Branch("falling_barycenter", &falling_barycenter, "falling_barycenter/s");
  signalTree->Branch("size", &size, "size/s");
  signalTree->Branch("maxSat", &maxSat, "maxSat/s");
  signalTree->Branch("charge", &charge, "charge/I");
  signalTree->Branch("max_adc", &max_adc, "max_adc/I");
  signalTree->Branch("max_adc_idx", &max_adc_idx, "max_adc_idx/I");
  signalTree->Branch("max_adc_x", &max_adc_x, "max_adc_x/F");
  signalTree->Branch("max_adc_y", &max_adc_y, "max_adc_y/F");
  signalTree->Branch("max_adc_z", &max_adc_z, "max_adc_z/F");
  signalTree->Branch("dr_min_pixelTrk", &dr_min_pixelTrk, "dr_min_pixelTrk/F");
  signalTree->Branch("adc_none", &adc_mone, "adc_mone/I");
  signalTree->Branch("adc_ntwo", &adc_mtwo, "adc_mtwo/I");
  signalTree->Branch("adc_nthree", &adc_mthree, "adc_mthree/I");
  signalTree->Branch("adc_nfour", &adc_mfour, "adc_mfour/I");
  signalTree->Branch("adc_pone", &adc_pone, "adc_pone/I");
  signalTree->Branch("adc_ptwo", &adc_ptwo, "adc_ptwo/I");
  signalTree->Branch("adc_pthree", &adc_pthree, "adc_pthree/I");
  signalTree->Branch("adc_pfour", &adc_pfour, "adc_pfour/I");
  signalTree->Branch("low_pt_trk_cluster", &low_pt_trk_cluster, "low_pt_trk_cluster/b");
  signalTree->Branch("high_pt_trk_cluster", &high_pt_trk_cluster, "high_pt_trk_cluster/b");
  signalTree->Branch("trk_algo", &trk_algo, "trk_algo/I");
  signalTree->Branch("n_saturated", &n_saturated, "n_saturated/I");
  signalTree->Branch("n_consecutive_saturated", &n_consecutive_saturated, "n_consecutive_saturated/I");
  signalTree->Branch("isTIB", &isTIB, "isTIB/O");
  signalTree->Branch("isTOB", &isTOB, "isTOB/O");
  signalTree->Branch("isTID", &isTID, "isTID/O");
  signalTree->Branch("isTEC", &isTEC, "isTEC/O");
  signalTree->Branch("isStereo", &isStereo, "isStereo/O");
  signalTree->Branch("isglued", &isglued, "isglued/O");
  signalTree->Branch("isstacked", &isstacked, "isstacked/O");
  signalTree->Branch("layer", &layer, "layer/i");
  
  signalTree->Branch("x", hitX, "x[size]/F");
  signalTree->Branch("y", hitY, "y[size]/F");
  signalTree->Branch("z", hitZ, "z[size]/F");
  signalTree->Branch("channel", channel, "channel[size]/s");
  signalTree->Branch("adc", adc, "adc[size]/s");
  signalTree->Branch("saturated", &saturated, "saturated/O");
  signalTree->Branch("diff_adc_pone", &diff_adc_pone, "diff_adc_pone/I");
  signalTree->Branch("diff_adc_ptwo", &diff_adc_ptwo, "diff_adc_ptwo/I");
  signalTree->Branch("diff_adc_pthree", &diff_adc_pthree, "diff_adc_pthree/I");
  signalTree->Branch("diff_adc_mone", &diff_adc_mone, "diff_adc_mone/I");
  signalTree->Branch("diff_adc_mtwo", &diff_adc_mtwo, "diff_adc_mtwo/I");
  signalTree->Branch("diff_adc_mthree", &diff_adc_mthree, "diff_adc_mthree/I");
  signalTree->Branch("noise_adc_pone", &noise_adc_pone, "noise_adc_pone/F");
  signalTree->Branch("noise_max_adc", &noise_max_adc, "noise_max_adc/F");
  signalTree->Branch("noise_adc_ptwo", &noise_adc_ptwo, "noise_adc_ptwo/F");
  signalTree->Branch("noise_adc_pthree", &noise_adc_pthree, "noise_adc_pthree/F");
  signalTree->Branch("noise_adc_mone", &noise_adc_mone, "noise_adc_mone/F");
  signalTree->Branch("noise_adc_mtwo", &noise_adc_mtwo, "noise_adc_mtwo/F");
  signalTree->Branch("noise_adc_mthree", &noise_adc_mthree, "noise_adc_mthree/F");
  signalTree->Branch("adc_std", &adc_std, "adc_std/F");
  signalTree->Branch("adc_zero", &adc_zero, "adc_zero/I");
  signalTree->Branch("adc_one", &adc_one, "adc_one/I");
  signalTree->Branch("adc_two", &adc_two, "adc_two/I");
  signalTree->Branch("adc_three", &adc_three, "adc_three/I");
  signalTree->Branch("adc_four", &adc_four, "adc_four/I");

  bkgTree = fs->make<TTree>("bkgTree", "bkgTree");
  bkgTree->Branch("event", &eventN, "event/i");
  bkgTree->Branch("run",   &runN, "run/I");
  bkgTree->Branch("lumi",  &lumi, "lumi/I");
  bkgTree->Branch("trk_pt", &trk_pt, "trk_pt/F");

  bkgTree->Branch("detId", &detId, "detId/i");
  bkgTree->Branch("firstStrip", &firstStrip, "firstStrip/s");
  bkgTree->Branch("endStrip", &endStrip, "endStrip/s");
  bkgTree->Branch("barycenter", &barycenter, "barycenter/F");
  bkgTree->Branch("vrtx_xy", &vrtx_xy, "vrtx_xy/F");
  bkgTree->Branch("vrtx_z", &vrtx_z, "vrtx_z/F");
  bkgTree->Branch("falling_barycenter", &falling_barycenter, "falling_barycenter/s");
  bkgTree->Branch("size", &size, "size/s");
  bkgTree->Branch("maxSat", &maxSat, "maxSat/s");
  bkgTree->Branch("charge", &charge, "charge/I");
  bkgTree->Branch("max_adc", &max_adc, "max_adc/I");
  bkgTree->Branch("max_adc_idx", &max_adc_idx, "max_adc_idx/I");
  bkgTree->Branch("max_adc_x", &max_adc_x, "max_adc_x/F");
  bkgTree->Branch("max_adc_y", &max_adc_y, "max_adc_y/F");
  bkgTree->Branch("max_adc_z", &max_adc_z, "max_adc_z/F");
  bkgTree->Branch("dr_min_pixelTrk", &dr_min_pixelTrk, "dr_min_pixelTrk/F");
  bkgTree->Branch("diff_adc_pone", &diff_adc_pone, "diff_adc_pone/I");
  bkgTree->Branch("diff_adc_ptwo", &diff_adc_ptwo, "diff_adc_ptwo/I");
  bkgTree->Branch("diff_adc_pthree", &diff_adc_pthree, "diff_adc_pthree/I");
  bkgTree->Branch("diff_adc_mone", &diff_adc_mone, "diff_adc_mone/I");
  bkgTree->Branch("diff_adc_mtwo", &diff_adc_mtwo, "diff_adc_mtwo/I");
  bkgTree->Branch("diff_adc_mthree", &diff_adc_mthree, "diff_adc_mthree/I");
  bkgTree->Branch("noise_max_adc", &noise_max_adc, "noise_max_adc/F");
  bkgTree->Branch("noise_adc_pone", &noise_adc_pone, "noise_adc_pone/F");
  bkgTree->Branch("noise_adc_ptwo", &noise_adc_ptwo, "noise_adc_ptwo/F");
  bkgTree->Branch("noise_adc_pthree", &noise_adc_pthree, "noise_adc_pthree/F");
  bkgTree->Branch("noise_adc_mone", &noise_adc_mone, "noise_adc_mone/F");
  bkgTree->Branch("noise_adc_mtwo", &noise_adc_mtwo, "noise_adc_mtwo/F");
  bkgTree->Branch("noise_adc_mthree", &noise_adc_mthree, "noise_adc_mthree/F");
  bkgTree->Branch("adc_none", &adc_mone, "adc_mone/I");
  bkgTree->Branch("adc_ntwo", &adc_mtwo, "adc_mtwo/I");
  bkgTree->Branch("adc_nthree", &adc_mthree, "adc_mthree/I");
  bkgTree->Branch("adc_nfour", &adc_mfour, "adc_mfour/I");
  bkgTree->Branch("adc_pone", &adc_pone, "adc_pone/I");
  bkgTree->Branch("adc_ptwo", &adc_ptwo, "adc_ptwo/I");
  bkgTree->Branch("adc_pthree", &adc_pthree, "adc_pthree/I");
  bkgTree->Branch("adc_pfour", &adc_pfour, "adc_pfour/I");
  bkgTree->Branch("adc_pfour", &adc_pfour, "adc_pfour/I");
  bkgTree->Branch("saturated", &saturated, "saturated/O");
  bkgTree->Branch("n_saturated", &n_saturated, "n_saturated/I");
  bkgTree->Branch("n_consecutive_saturated", &n_consecutive_saturated, "n_consecutive_saturated/I");
  bkgTree->Branch("adc_std", &adc_std, "adc_std/F");
  bkgTree->Branch("adc_zero", &adc_zero, "adc_zero/I");
  bkgTree->Branch("adc_one", &adc_one, "adc_one/I");
  bkgTree->Branch("adc_two", &adc_two, "adc_two/I");
  bkgTree->Branch("adc_three", &adc_three, "adc_three/I");
  bkgTree->Branch("adc_four", &adc_four, "adc_four/I");
  bkgTree->Branch("isTIB", &isTIB, "isTIB/O");
  bkgTree->Branch("isTOB", &isTOB, "isTOB/O");
  bkgTree->Branch("isTID", &isTID, "isTID/O");
  bkgTree->Branch("isTEC", &isTEC, "isTEC/O");
  bkgTree->Branch("isStereo", &isStereo, "isStereo/O");
  bkgTree->Branch("isglued", &isglued, "isglued/O");
  bkgTree->Branch("isstacked", &isstacked, "isstacked/O");
  bkgTree->Branch("layer", &layer, "layer/i");

  bkgTree->Branch("x", hitX, "x[size]/F");
  bkgTree->Branch("y", hitY, "y[size]/F");
  bkgTree->Branch("z", hitZ, "z[size]/F");
  bkgTree->Branch("channel", channel, "channel[size]/s");
  bkgTree->Branch("adc", adc, "adc[size]/s");

}

nn_tupleProducer_raw::~nn_tupleProducer_raw() = default;

void nn_tupleProducer_raw::analyze(const edm::Event& event, const edm::EventSetup& es) {
  edm::Handle<edmNew::DetSetVector<SiStripCluster>> clusterCollection = event.getHandle(clusterToken);
  const auto& tracksHandle = event.getHandle(tracksToken_);
  const auto& hlttracksHandle = event.getHandle(hlttracksToken_);
  const auto* stripCPE    = &es.getData(stripCPEToken_);
  const auto* magField    = &es.getData(magFieldToken_);

  const auto& theNoise_ = &es.getData(stripNoiseToken_);

  const Propagator* thePropagator = &es.getData(propagatorToken_);

  theTTrackBuilder = &es.getData(ttbToken_);
  using namespace edm;

  const auto& vertexHandle = event.getHandle(vertexToken_);

  if (!tracksHandle.isValid()) {
    edm::LogError("flatNtuple_producer") << "No valid track collection found";
    return;
  }

  if (!hlttracksHandle.isValid()) {
	      edm::LogError("flatNtuple_producer") << "No valid track collection found";
	          return;
  }
  
  if (!vertexHandle.isValid()) {
	      edm::LogError("flatNtuple_producer") << "No valid vertex collection found";
	   return;
  }

  const reco::TrackCollection& tracks = *tracksHandle;
  const reco::TrackCollection& hlttracks = *hlttracksHandle;
  const reco::VertexCollection vertices = *vertexHandle;
  //if ( tracks.size() != 1 || event.id().event() !=33) return;
  //std::cout << "event " << event.id().event() << std::endl;
  std::map<uint32_t, std::vector<cluster_property>> matched_cluster;
  //int trkcluster = 0;
  for(unsigned int i=0; i<tracks.size(); i++) {
     auto trk = tracks.at(i);
     for (auto ih = trk.recHitsBegin(); ih != trk.recHitsEnd(); ih++) {
         const SiStripCluster* strip=NULL;
         const TrackingRecHit& hit = **ih;
         const DetId detId((hit).geographicalId());
         if (detId.det() == DetId::Tracker) { 
           if (detId.subdetId() == kBPIX || detId.subdetId() == kFPIX) continue;  // pixel is always 2D
           else {        // should be SiStrip now
		   //std::cout << "subdet " << detId.subdetId() << std::endl;
               if (dynamic_cast<const SiStripRecHit1D *>(&hit)) {
		   //std::cout << " found SiStripRecHit1D " << std::endl;
                   strip = dynamic_cast<const SiStripRecHit1D *>(&hit)->cluster().get();
               }
               else if ( dynamic_cast<const SiStripRecHit2D *>(&hit)) {
                 //std::cout << "found SiStripRecHit2D " << std::endl;
                 strip = dynamic_cast<const SiStripRecHit2D *>(&hit)->cluster().get();
               }
               else if (dynamic_cast<const SiStripMatchedRecHit2D *>(&hit)) {
                 // std::cout << "found SiStripMatchedRecHit2D " << std::endl;
                  strip = &(dynamic_cast<const SiStripMatchedRecHit2D *>(&hit))->monoCluster();
              }
           }
           if(strip) {
		   //trkcluster += 1;
	       //std::cout << "strip " << strip << std::endl;
               bool low_pt_trk = trk.pt() < 1.;
               matched_cluster[detId].emplace_back(
                      low_pt_trk, !low_pt_trk, strip->barycenter(),
                      strip->size(), strip->firstStrip(), strip->endStrip(),
                      strip->charge(),
                      trk.algo()
               );
         }
        }
    }
  }    
  const auto& tkGeom = &es.getData(tkGeomToken_);
  const auto tkDets = tkGeom->dets();
  const TrackerTopology& tTopo = es.getData(tTopoToken_);

  //const auto& theFilter = &es.getData(csfToken_);
  //int cluster = 0;
  for (const auto& detSiStripClusters : *clusterCollection) {
    isTIB = isTOB = isTID = isTEC = isStereo = 0;
    eventN = event.id().event();
    //if (eventN != 24061779) continue;
    runN   = (int) event.id().run();
    lumi   = (int) event.id().luminosityBlock();
    detId  = detSiStripClusters.id();
    SiStripNoises::Range detNoiseRange = theNoise_->getRange(detId);
    uint32_t subdet = DetId(detId).subdetId();
    //std::cout << "2nd subdet " << subdet << std::endl;
    if (subdet == SiStripSubdetector::TIB) isTIB = 1;
    else if (subdet == SiStripSubdetector::TOB) isTOB = 1;
    else if (subdet == SiStripSubdetector::TID) isTID = 1;
    else if (subdet == SiStripSubdetector::TEC) isTEC = 1;
    layer = tTopo.layer(DetId(detId));
    isStereo = tTopo.isStereo(DetId(detId));
    isglued = tTopo.glued(DetId(detId));
    isstacked = tTopo.stack(DetId(detId));
    const auto& _detId = detId; // for the capture clause in the lambda function
    auto det = std::find_if(tkDets.begin(), tkDets.end(), [_detId](auto& elem) -> bool {
        return (elem->geographicalId().rawId() == _detId);
    });
    const StripTopology& p = dynamic_cast<const StripGeomDetUnit*>(*det)->specificTopology();
    std::vector<cluster_property> track_clusters = {};
    if ( matched_cluster.find(detId) != matched_cluster.end() ) track_clusters = matched_cluster[detId];

    for (const auto& stripCluster : detSiStripClusters) {
    //  cluster += 1;
      bool signal = 0;
      firstStrip = stripCluster.firstStrip();
      endStrip   = stripCluster.endStrip();
      barycenter = stripCluster.barycenter();
      size       = stripCluster.size();
      charge     = stripCluster.charge();

      max_adc_idx = 0;
      max_adc_x = max_adc_y = max_adc_z = 0.;
      max_adc = 0;
      adc_mone = 0;
      adc_mtwo = 0;
      adc_mthree = 0;
      adc_mfour = 0;
      adc_pone = 0;
      adc_ptwo = 0;
      adc_pthree = 0;
      adc_pfour = 0;
      diff_adc_mone = diff_adc_mtwo = diff_adc_mthree = diff_adc_pone = diff_adc_ptwo = diff_adc_pthree = n_saturated = n_consecutive_saturated = adc_zero = adc_one = adc_two = adc_three = adc_four = noise_adc_mone = noise_adc_mtwo = noise_adc_mthree = noise_adc_pone = noise_adc_ptwo = noise_adc_pthree = 0;
      vrtx_xy = 99;
      vrtx_z = 99;
      trk_pt = 1000;
      std::vector<int>adcs;
      std::vector<float>noises;
      //uint16_t thisSat = maxSat = 0;
      for (int strip = firstStrip; strip < endStrip; ++strip)
      {
        GlobalPoint gp = (tkGeom->idToDet(detId))->surface().toGlobal(p.localPosition((float) strip));

        hitX   [strip - firstStrip] = gp.x();
        hitY   [strip - firstStrip] = gp.y();
        hitZ   [strip - firstStrip] = gp.z();
        channel[strip - firstStrip] = strip;
        adc    [strip - firstStrip] = stripCluster[strip - firstStrip];
	if ((strip - firstStrip) == 0 ) adc_zero = stripCluster[strip - firstStrip];
	else if ((strip - firstStrip) == 1 ) adc_one = stripCluster[strip - firstStrip];
	else if ((strip - firstStrip) == 2 ) adc_two = stripCluster[strip - firstStrip];
	else if ((strip - firstStrip) == 3 ) adc_three = stripCluster[strip - firstStrip];
	else if ((strip - firstStrip) == 4 ) adc_four = stripCluster[strip - firstStrip];
	adcs.push_back(stripCluster[strip - firstStrip]);
	noises.push_back(theNoise_->getNoise(strip, detNoiseRange));
	//if ( int(stripCluster[strip - firstStrip]) - theNoise_->getNoise(strip, detNoiseRange) < 0) 
	//std::cout << "noise " << strip << "\t" << firstStrip << "\t" << endStrip << "\t" << int(stripCluster[strip - firstStrip]) << "\t" << theNoise_->getNoise(strip, detNoiseRange) << std::endl; 
	if (adc    [strip - firstStrip] >= 254) n_saturated += 1;
        if (adc [strip - firstStrip] > max_adc) {
                max_adc = adc[strip - firstStrip];
                max_adc_idx = strip - firstStrip;
		max_adc_x   = hitX[strip - firstStrip];
		max_adc_y   = hitY[strip - firstStrip];
		max_adc_z   = hitZ[strip - firstStrip];
        }	
      }
      double mean = std::accumulate(adc, adc+size,0.0) / size;
      adc_std = std::sqrt(std::accumulate(adc, adc+size,0.0, [mean](double acc, double x) {
			      return acc + (x - mean) * (x - mean);
			      }) / size );
      noise_max_adc = adc[max_adc_idx] - noises[max_adc_idx];
      if (max_adc_idx >=1) {
	      diff_adc_mone = adc[max_adc_idx] - adc[max_adc_idx-1];
	      //noise_diff_adc_mone = abs(adc[max_adc_idx-1] - noises[max_adc_idx-1]);
	      noise_adc_mone = noises[max_adc_idx-1];
	      //if (noise_diff_adc_mone <0) std::cout << "negetive " << max_adc_idx << "\t" << firstStrip << "\t" << endStrip << "\t" << size << std::endl;
	      adc_mone = adc[max_adc_idx-1];
      }
      if (max_adc_idx >=2) {
	      diff_adc_mtwo = adc[max_adc_idx] - adc[max_adc_idx-2];
	      //noise_diff_adc_mtwo = abs(adc[max_adc_idx-2] - noises[max_adc_idx-2]);
              noise_adc_mtwo = noises[max_adc_idx-2];
	      //if (noise_diff_adc_mtwo <0) std::cout << "negetive mone " << max_adc_idx << "\t" << firstStrip << "\t" << endStrip << "\t" << size << std::endl;
	      adc_mtwo = adc[max_adc_idx-2];
      }
      if (max_adc_idx >=3) {
              diff_adc_mthree = adc[max_adc_idx] - adc[max_adc_idx-3];
	      //noise_diff_adc_mthree = abs(adc[max_adc_idx-3] - noises[max_adc_idx-3]);
	      noise_adc_mthree = noises[max_adc_idx-3];
	      //if (noise_diff_adc_mthree <0) std::cout << "negetive three " << max_adc_idx << "\t" << firstStrip << "\t" << endStrip << "\t" << size << std::endl;
              adc_mthree = adc[max_adc_idx-3];
      }
      if (((size-1)-max_adc_idx) >=1) {
	      diff_adc_pone = adc[max_adc_idx] - adc[max_adc_idx+1];
	      //noise_diff_adc_pone = abs(adc[max_adc_idx+1] - noises[max_adc_idx+1]);
	      noise_adc_pone = noises[max_adc_idx+1];
	      //if (noise_diff_adc_pone <0) std::cout << "negetive pone " << max_adc_idx << "\t" << firstStrip << "\t" << endStrip << "\t" << size << std::endl;
	      adc_pone = adc[max_adc_idx+1];
      }
      if (((size-1)-max_adc_idx) >=2) {
	      diff_adc_ptwo = adc[max_adc_idx] - adc[max_adc_idx+2];
	      //noise_diff_adc_ptwo = abs(adc[max_adc_idx+2] - noises[max_adc_idx+2]);
	      noise_adc_ptwo = noises[max_adc_idx+2];
	      //if (noise_diff_adc_ptwo <0) std::cout << "negetive ptwo " << max_adc_idx << "\t" << firstStrip << "\t" << endStrip << "\t" << size << std::endl;
	      adc_ptwo = adc[max_adc_idx+2];
      }
      if (((size-1)-max_adc_idx) >=3) {
              diff_adc_pthree = adc[max_adc_idx] - adc[max_adc_idx+3];
	      //noise_diff_adc_pthree = abs(adc[max_adc_idx+3] - noises[max_adc_idx+3]);
	      noise_adc_pthree = noises[max_adc_idx+3];
	      //if (noise_diff_adc_pthree <0) std::cout << "negetive pthree " << max_adc_idx << "\t" << firstStrip << "\t" << endStrip << "\t" << size << std::endl;
              adc_pthree = adc[max_adc_idx+3];
      }
      
      //mimicing the algorithm used in StripSubClusterShapeTrajectoryFilter...
      //  //Looks for 3 adjacent saturated strips (ADC>=254)
      const auto& ampls = stripCluster.amplitudes();
      unsigned int thisSat = (ampls[0] >= 254), maxSat = thisSat;
      for (unsigned int i = 1, n = ampls.size(); i < n; ++i) {
         if (ampls[i] >= 254) {
             thisSat++;
         } else if (thisSat > 0) {
             maxSat = std::max<int>(maxSat, thisSat);
             thisSat = 0;
            }
      }
      if (thisSat > 0) {
         maxSat = std::max<int>(maxSat, thisSat);
      }
      
      for(auto& trk_cluster_property: track_clusters)
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
	       signal = 1;
	       break;
           }
        }

      dr_min_pixelTrk = 99.;
      const GeomDetUnit* geomDet = tkGeom->idToDetUnit(detId);
      const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(geomDet);
      const reco::Track* recotrk = NULL;
      for ( const auto & trk : tracks) {
              if (!stripDet) continue;
              for (auto const& hit : trk.recHits()) {
                if (!hit->isValid()) continue;
                if (hit->geographicalId() != detId) continue;
                reco::TransientTrack tkTT = theTTrackBuilder->build(trk);
                // Propagate track to the cluster surface
                TrajectoryStateOnSurface tsos = thePropagator->propagate(tkTT.innermostMeasurementState(), geomDet->surface());
                if (!tsos.isValid()) continue;
                auto localValues = stripCPE->localParameters(stripCluster, *stripDet, tsos);
                LocalPoint clusterLocal = localValues.first;

                // Track position at same surface
                LocalPoint trackLocal = geomDet->surface().toLocal(tsos.globalPosition());
                //LocalPoint trackLocal = hit->localPosition();
                //LocalPoint clusterLocal = p.localPosition(stripCluster.barycenter());
                float dr = abs(trackLocal.x() - clusterLocal.x()); //std::sqrt( pow( (trackLocal.x() - clusterLocal.x()), 2 ) +
                if (dr < dr_min_pixelTrk) {
                  dr_min_pixelTrk = dr;
		  recotrk = &trk;
                  trk_pt = trk.pt();
                }
              }
      }

      const reco::Track* hlttrk = NULL;
      for ( const auto & trk : hlttracks) {
	      vrtx_xy = trk.dxy(vertices.at(0).position());
	      vrtx_z = trk.dz(vertices.at(0).position());
	      if (!stripDet) continue;
	      for (auto const& hit : trk.recHits()) {
                if (!hit->isValid()) continue;
                if (hit->geographicalId() != detId) continue;
		reco::TransientTrack tkTT = theTTrackBuilder->build(trk);
		// Propagate track to the cluster surface
                TrajectoryStateOnSurface tsos = thePropagator->propagate(tkTT.innermostMeasurementState(), geomDet->surface());
                if (!tsos.isValid()) continue;
                auto localValues = stripCPE->localParameters(stripCluster, *stripDet, tsos);
                LocalPoint clusterLocal = localValues.first;

                // Track position at same surface
                LocalPoint trackLocal = geomDet->surface().toLocal(tsos.globalPosition());
	        //LocalPoint trackLocal = hit->localPosition();
                //LocalPoint clusterLocal = p.localPosition(stripCluster.barycenter());
                float dr = abs(trackLocal.x() - clusterLocal.x()); //std::sqrt( pow( (trackLocal.x() - clusterLocal.x()), 2 ) +
                if (dr < dr_min_pixelTrk) {
                  dr_min_pixelTrk = dr;
		  hlttrk = &trk;
		  trk_pt = trk.pt();
                }
	      }
      }
      if (signal) {
	   if (hlttrk && recotrk) {
		   std::cout << "sig pt " << hlttrk->pt() << "\t" << recotrk->pt() << std::endl;
	           std::cout << "sig pt " << vrtx_xy << "\t" << recotrk->dxy(vertices.at(0).position()) << "\t" << vrtx_z << "\t" << recotrk->dz(vertices.at(0).position()) << std::endl;
	   }

	      else if (!hlttrk) { std::cout << "sig not found hlt" << std::endl;}
	      else if (!recotrk) { std::cout << "sig not found reco" << std::endl;}
	      signalTree->Fill();
	      for (unsigned int i = 0; i < adcs.size()-1; ++i) hist_sig->Fill(i, adcs[i]); // Fill 2D histogram
      }
      else {
	      bkgTree->Fill();
	      for (unsigned int i = 0; i < adcs.size()-1; ++i) hist_bkg->Fill(i, adcs[i]); // Fill 2D histogram
      }
    }
  }
}

void nn_tupleProducer_raw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("siStripClustersTag", edm::InputTag("siStripClusters"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks","","reRECO"));
  desc.add<edm::InputTag>("hlttracks", edm::InputTag("hltTracks","","HLTX"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("vertex", edm::InputTag("vertex"));
  descriptions.add("nn_tupleProducer_raw", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(nn_tupleProducer_raw);
