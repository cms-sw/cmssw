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
  edm::EDGetTokenT<reco::TrackCollection> hltPixeltracksToken_;

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

  TTree* tree;
  TTree* regression;
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
  float       pixeltrk_dr_min;
  float       hlttrk_dr_min;
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
  float       ref_hitY[nMax];
  uint16_t    ref_channel[nMax];
  uint16_t    ref_adc[nMax];
  float cluster_x;
  float cluster_y;
  float cluster_z;
  float recotrk_dr_min;
  float dr;
  float pixeltrk_pt;
  float pixeltrk_pterr;
  float pixeltrk_eta;
  float pixeltrk_phi;
  float pixeltrk_dz;
  float pixeltrk_dxy;
  int pixeltrk_validhits;
  float pixeltrk_chi2;
  float pixeltrk_d0sigma;
  float pixeltrk_dzsigma;
  float pixeltrk_qoverp;
  float pixeltrk_qoverperror;

  float hlttrk_pt;
  float hlttrk_pterr;
  float hlttrk_eta;
  float hlttrk_phi;
  float hlttrk_dz;
  float hlttrk_dxy;
  int   hlttrk_validhits;
  float hlttrk_chi2;
  float hlttrk_d0sigma;
  float hlttrk_dzsigma;
  float hlttrk_qoverp;
  float hlttrk_qoverperror;

  float recotrk_pt;
  float recotrk_pterr;
  float recotrk_eta;
  float recotrk_phi;
  float recotrk_dz;
  float recotrk_dxy;
  int   recotrk_validhits;
  float recotrk_chi2;
  float recotrk_d0sigma;
  float recotrk_dzsigma;
  float recotrk_qoverp;
  float recotrk_qoverperror;

  bool target;
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
  hltPixeltracksToken_           = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("hltPixeltracks"));
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
  tree = fs->make<TTree>("tree", "tree");
  regression = fs->make<TTree>("regression", "regression");

  regression->Branch("event", &eventN, "event/i");
  regression->Branch("run",   &runN, "run/I");
  regression->Branch("lumi",  &lumi, "lumi/I");
  regression->Branch("pixeltrk_pt", &pixeltrk_pt, "pixeltrk_pt/F");
  regression->Branch("pixeltrk_pterr", &pixeltrk_pterr, "pixeltrk_pterr/F");
  regression->Branch("pixeltrk_eta", &pixeltrk_eta, "pixeltrk_eta/F");
  regression->Branch("pixeltrk_phi", &pixeltrk_phi, "pixeltrk_phi/F");
  regression->Branch("pixeltrk_dz", &pixeltrk_dz, "pixeltrk_dz/F");
  regression->Branch("pixeltrk_dxy", &pixeltrk_dxy, "pixeltrk_dxy/F");
  regression->Branch("cluster_x", &cluster_x, "cluster_x/F");
  regression->Branch("cluster_y", &cluster_y, "cluster_y/F");
  regression->Branch("cluster_z", &cluster_z, "cluster_z/F");
  regression->Branch("target", &target, "target/O");
  regression->Branch("recotrk_dr_min", &recotrk_dr_min, "recotrk_dr_min/F");
  regression->Branch("dr", &dr, "dr/F");
  regression->Branch("pixeltrk_dr_min", &pixeltrk_dr_min, "pixeltrk_dr_min/F");
  regression->Branch("pixeltrk_validhits", &pixeltrk_validhits, "pixeltrk_validhits/I");
  regression->Branch("pixeltrk_chi2", &pixeltrk_chi2, "pixeltrk_chi2/F");
  regression->Branch("pixeltrk_d0sigma", &pixeltrk_d0sigma, "pixeltrk_d0sigma/F");
  regression->Branch("pixeltrk_dzsigma", &pixeltrk_dzsigma, "pixeltrk_dzsigma/F");
  regression->Branch("pixeltrk_qoverp", &pixeltrk_qoverp, "pixeltrk_qoverp/F");
  regression->Branch("pixeltrk_qoverperror", &pixeltrk_qoverperror, "pixeltrk_qoverperror/F");

  tree->Branch("cluster_x", &cluster_x, "cluster_x/F");
  tree->Branch("cluster_y", &cluster_y, "cluster_y/F");
  tree->Branch("cluster_z", &cluster_z, "cluster_z/F");
  tree->Branch("hlttrk_pt", &hlttrk_pt, "hlttrk_pt/F");
  tree->Branch("hlttrk_pterr", &hlttrk_pterr, "hlttrk_pterr/F");
  tree->Branch("hlttrk_eta", &hlttrk_eta, "hlttrk_eta/F");
  tree->Branch("hlttrk_phi", &hlttrk_phi, "hlttrk_phi/F");
  tree->Branch("hlttrk_dz", &hlttrk_dz, "hlttrk_dz/F");
  tree->Branch("hlttrk_dxy", &hlttrk_dxy, "hlttrk_dxy/F");
  tree->Branch("hlttrk_validhits", &hlttrk_validhits, "hlttrk_validhits/I");
  tree->Branch("hlttrk_chi2", &hlttrk_chi2, "hlttrk_chi2/F");
  tree->Branch("hlttrk_d0sigma", &hlttrk_d0sigma, "hlttrk_d0sigma/F");
  tree->Branch("hlttrk_dzsigma", &hlttrk_dzsigma, "hlttrk_dzsigma/F");
  tree->Branch("hlttrk_qoverp", &hlttrk_qoverp, "hlttrk_qoverp/F");
  tree->Branch("hlttrk_qoverperror", &hlttrk_qoverperror, "hlttrk_qoverperror/F");
  tree->Branch("recotrk_pt", &recotrk_pt, "recotrk_pt/F");
  tree->Branch("recotrk_pterr", &recotrk_pterr, "recotrk_pterr/F");
  tree->Branch("recotrk_eta", &recotrk_eta, "recotrk_eta/F");
  tree->Branch("recotrk_phi", &recotrk_phi, "recotrk_phi/F");
  tree->Branch("recotrk_dz", &recotrk_dz, "recotrk_dz/F");
  tree->Branch("recotrk_dxy", &recotrk_dxy, "recotrk_dxy/F");
  tree->Branch("recotrk_validhits", &recotrk_validhits, "recotrk_validhits/I");
  tree->Branch("recotrk_chi2", &recotrk_chi2, "recotrk_chi2/F");
  tree->Branch("recotrk_d0sigma", &recotrk_d0sigma, "recotrk_d0sigma/F");
  tree->Branch("recotrk_dzsigma", &recotrk_dzsigma, "recotrk_dzsigma/F");
  tree->Branch("recotrk_qoverp", &recotrk_qoverp, "recotrk_qoverp/F");
  tree->Branch("recotrk_qoverperror", &recotrk_qoverperror, "recotrk_qoverperror/F");
  tree->Branch("target", &target, "target/O");
  tree->Branch("recotrk_dr_min", &recotrk_dr_min, "recotrk_dr_min/F");
  tree->Branch("dr", &dr, "dr/F");
  tree->Branch("pixeltrk_dr_min", &pixeltrk_dr_min, "pixeltrk_dr_min/F");
  tree->Branch("pixeltrk_validhits", &pixeltrk_validhits, "pixeltrk_validhits/I");
  tree->Branch("pixeltrk_chi2", &pixeltrk_chi2, "pixeltrk_chi2/F");
  tree->Branch("pixeltrk_d0sigma", &pixeltrk_d0sigma, "pixeltrk_d0sigma/F");
  tree->Branch("pixeltrk_dzsigma", &pixeltrk_dzsigma, "pixeltrk_dzsigma/F");
  tree->Branch("pixeltrk_qoverp", &pixeltrk_qoverp, "pixeltrk_qoverp/F");
  tree->Branch("pixeltrk_qoverperror", &pixeltrk_qoverperror, "pixeltrk_qoverperror/F");
  tree->Branch("pixeltrk_pt", &pixeltrk_pt, "pixeltrk_pt/F");
  tree->Branch("pixeltrk_pterr", &pixeltrk_pterr, "pixeltrk_pterr/F");
  tree->Branch("pixeltrk_eta", &pixeltrk_eta, "pixeltrk_eta/F");
  tree->Branch("pixeltrk_phi", &pixeltrk_phi, "pixeltrk_phi/F");
  tree->Branch("pixeltrk_dz", &pixeltrk_dz, "pixeltrk_dz/F");
  tree->Branch("pixeltrk_dxy", &pixeltrk_dxy, "pixeltrk_dxy/F");
  tree->Branch("event", &eventN, "event/i");
  tree->Branch("run",   &runN, "run/I");
  tree->Branch("lumi",  &lumi, "lumi/I");

  tree->Branch("detId", &detId, "detId/i");
  tree->Branch("firstStrip", &firstStrip, "firstStrip/s");
  tree->Branch("endStrip", &endStrip, "endStrip/s");
  tree->Branch("barycenter", &barycenter, "barycenter/F");
  tree->Branch("vrtx_xy", &vrtx_xy, "vrtx_xy/F");
  tree->Branch("vrtx_z", &vrtx_z, "vrtx_z/F");
  tree->Branch("falling_barycenter", &falling_barycenter, "falling_barycenter/s");
  tree->Branch("size", &size, "size/s");
  tree->Branch("maxSat", &maxSat, "maxSat/s");
  tree->Branch("charge", &charge, "charge/I");
  tree->Branch("max_adc", &max_adc, "max_adc/I");
  tree->Branch("max_adc_idx", &max_adc_idx, "max_adc_idx/I");
  tree->Branch("max_adc_x", &max_adc_x, "max_adc_x/F");
  tree->Branch("max_adc_y", &max_adc_y, "max_adc_y/F");
  tree->Branch("max_adc_z", &max_adc_z, "max_adc_z/F");
  tree->Branch("pixeltrk_dr_min", &pixeltrk_dr_min, "pixeltrk_dr_min/F");
  tree->Branch("hlttrk_dr_min", &hlttrk_dr_min, "hlttrk_dr_min/F");
  tree->Branch("adc_none", &adc_mone, "adc_mone/I");
  tree->Branch("adc_ntwo", &adc_mtwo, "adc_mtwo/I");
  tree->Branch("adc_nthree", &adc_mthree, "adc_mthree/I");
  tree->Branch("adc_nfour", &adc_mfour, "adc_mfour/I");
  tree->Branch("adc_pone", &adc_pone, "adc_pone/I");
  tree->Branch("adc_ptwo", &adc_ptwo, "adc_ptwo/I");
  tree->Branch("adc_pthree", &adc_pthree, "adc_pthree/I");
  tree->Branch("adc_pfour", &adc_pfour, "adc_pfour/I");
  tree->Branch("low_pt_trk_cluster", &low_pt_trk_cluster, "low_pt_trk_cluster/b");
  tree->Branch("high_pt_trk_cluster", &high_pt_trk_cluster, "high_pt_trk_cluster/b");
  tree->Branch("trk_algo", &trk_algo, "trk_algo/I");
  tree->Branch("n_saturated", &n_saturated, "n_saturated/I");
  tree->Branch("n_consecutive_saturated", &n_consecutive_saturated, "n_consecutive_saturated/I");
  tree->Branch("isTIB", &isTIB, "isTIB/O");
  tree->Branch("isTOB", &isTOB, "isTOB/O");
  tree->Branch("isTID", &isTID, "isTID/O");
  tree->Branch("isTEC", &isTEC, "isTEC/O");
  tree->Branch("isStereo", &isStereo, "isStereo/O");
  tree->Branch("isglued", &isglued, "isglued/O");
  tree->Branch("isstacked", &isstacked, "isstacked/O");
  tree->Branch("layer", &layer, "layer/i");
  
  tree->Branch("x", hitX, "x[size]/F");
  tree->Branch("y", hitY, "y[size]/F");
  tree->Branch("z", hitZ, "z[size]/F");
  tree->Branch("channel", channel, "channel[size]/s");
  tree->Branch("adc", adc, "adc[size]/s");
  tree->Branch("saturated", &saturated, "saturated/O");
  tree->Branch("diff_adc_pone", &diff_adc_pone, "diff_adc_pone/I");
  tree->Branch("diff_adc_ptwo", &diff_adc_ptwo, "diff_adc_ptwo/I");
  tree->Branch("diff_adc_pthree", &diff_adc_pthree, "diff_adc_pthree/I");
  tree->Branch("diff_adc_mone", &diff_adc_mone, "diff_adc_mone/I");
  tree->Branch("diff_adc_mtwo", &diff_adc_mtwo, "diff_adc_mtwo/I");
  tree->Branch("diff_adc_mthree", &diff_adc_mthree, "diff_adc_mthree/I");
  tree->Branch("noise_adc_pone", &noise_adc_pone, "noise_adc_pone/F");
  tree->Branch("noise_max_adc", &noise_max_adc, "noise_max_adc/F");
  tree->Branch("noise_adc_ptwo", &noise_adc_ptwo, "noise_adc_ptwo/F");
  tree->Branch("noise_adc_pthree", &noise_adc_pthree, "noise_adc_pthree/F");
  tree->Branch("noise_adc_mone", &noise_adc_mone, "noise_adc_mone/F");
  tree->Branch("noise_adc_mtwo", &noise_adc_mtwo, "noise_adc_mtwo/F");
  tree->Branch("noise_adc_mthree", &noise_adc_mthree, "noise_adc_mthree/F");
  tree->Branch("adc_std", &adc_std, "adc_std/F");
  tree->Branch("adc_zero", &adc_zero, "adc_zero/I");
  tree->Branch("adc_one", &adc_one, "adc_one/I");
  tree->Branch("adc_two", &adc_two, "adc_two/I");
  tree->Branch("adc_three", &adc_three, "adc_three/I");
  tree->Branch("adc_four", &adc_four, "adc_four/I");

}

nn_tupleProducer_raw::~nn_tupleProducer_raw() = default;

void nn_tupleProducer_raw::analyze(const edm::Event& event, const edm::EventSetup& es) {
  edm::Handle<edmNew::DetSetVector<SiStripCluster>> clusterCollection = event.getHandle(clusterToken);
  const auto& tracksHandle = event.getHandle(tracksToken_);
  const auto& hlttracksHandle = event.getHandle(hlttracksToken_);
  const auto& hltPixeltracksHandle = event.getHandle(hltPixeltracksToken_);
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
  if (!hltPixeltracksHandle.isValid()) {
	edm::LogError("flatNtuple_producer") << "No valid track collection found";
	return;
  }
  
  if (!vertexHandle.isValid()) {
	      edm::LogError("flatNtuple_producer") << "No valid vertex collection found";
	   return;
  }

  const reco::TrackCollection& tracks = *tracksHandle;
  const reco::TrackCollection& hlttracks = *hlttracksHandle;
  const reco::TrackCollection& hltPixeltracks = *hltPixeltracksHandle;
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
      target = 0;
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
      pixeltrk_pt = 0;
      pixeltrk_eta = 5;
      pixeltrk_phi = 5;
      pixeltrk_qoverperror = 2;
      pixeltrk_validhits = pixeltrk_qoverp = 0;
      pixeltrk_pterr = 2;
      pixeltrk_chi2 = 1000;
      pixeltrk_dz = 50;
      pixeltrk_dzsigma = 50;
      pixeltrk_dxy = 2;
      pixeltrk_d0sigma = 50;

      hlttrk_pt = 0;
      hlttrk_eta = 5;
      hlttrk_phi = 5;
      hlttrk_qoverperror = 2;
      hlttrk_validhits = hlttrk_qoverp = 0;
      hlttrk_pterr = 2;
      hlttrk_chi2 = 1000;
      hlttrk_dz = 50;
      hlttrk_dzsigma = 50;
      hlttrk_dxy = 2;
      hlttrk_d0sigma = 50;

      recotrk_pt = 0;
      recotrk_eta = 5;
      recotrk_phi = 5;
      recotrk_qoverperror = 2;
      recotrk_validhits = recotrk_qoverp = 0;
      recotrk_pterr = 2;
      recotrk_chi2 = 1000;
      recotrk_dz = 50;
      recotrk_dzsigma = 50;
      recotrk_dxy = 2;
      recotrk_d0sigma = 50;

      vrtx_xy = 99;
      vrtx_z = 99;
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
	       target = 1;
	       break;
           }
        }

      pixeltrk_dr_min = 99.;
      const GeomDetUnit* geomDet = tkGeom->idToDetUnit(detId);
      const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(geomDet);
      
      const reco::Track* recotrk = NULL;
      auto clusterpositions = (tkGeom->idToDet(detId))->surface().toGlobal(stripCPE->localParameters(stripCluster, *stripDet).first);
      cluster_x = clusterpositions.x();
      cluster_y = clusterpositions.y();
      cluster_z = clusterpositions.z();
      hlttrk_dr_min = 99;
      recotrk_dr_min = 99;
      TrajectoryStateOnSurface tsos_pixel, tsos_reco;
      const reco::Track* pixeltrk = NULL;
      for ( const auto & trk : hltPixeltracks) {
            reco::TransientTrack tkTT = theTTrackBuilder->build(trk);
	    if(!tkTT.impactPointState().isValid()) continue;
            TrajectoryStateOnSurface tsos = thePropagator->propagate(tkTT.impactPointState(), geomDet->surface());
             if (!tsos.isValid()) continue;
             auto localValues = stripCPE->localParameters(stripCluster, *stripDet, tsos);
             LocalPoint clusterLocal = localValues.first;
             LocalPoint trackLocal = geomDet->surface().toLocal(tsos.globalPosition());
             float dr = abs(trackLocal.x() - clusterLocal.x());
             if (dr < pixeltrk_dr_min) {
                  pixeltrk_dr_min = dr;
		  tsos_pixel = tsos;
		  pixeltrk = &trk;
             }
      }
     
      for ( const auto & trk : tracks) {
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
                if (dr < recotrk_dr_min) {
                  recotrk_dr_min = dr;
		  recotrk = &trk;
		  tsos_reco = tsos;
                }
              }
      }

      if (tsos_reco.isValid()) {
        recotrk_pt = recotrk->pt();
        recotrk_pterr = recotrk->ptError();
        recotrk_eta = recotrk->eta();
        recotrk_phi = recotrk->phi();
        recotrk_dz = recotrk->dz(vertices.at(0).position());
        recotrk_dxy = recotrk->dxy(vertices.at(0).position());
        recotrk_validhits = recotrk->numberOfValidHits();
        recotrk_chi2 = recotrk->normalizedChi2();
        recotrk_d0sigma = sqrt(recotrk->d0Error() * recotrk->d0Error() + vertices.at(0).xError() * vertices.at(0).yError());
        recotrk_dzsigma = sqrt(recotrk->dzError() * recotrk->dzError() + vertices.at(0).zError() * vertices.at(0).zError());
        recotrk_qoverp = recotrk->qoverp();
        recotrk_qoverperror = recotrk->qoverpError();
      }
      /*dr = 99;
      if (tsos_pixel.isValid() && tsos_reco.isValid()) {
	      float deta = tsos_pixel.globalPosition().eta() - tsos_reco.globalPosition().eta();
	      float dphi = tsos_pixel.globalPosition().phi() - tsos_reco.globalPosition().phi();
              dr = std::sqrt( pow(deta,2) + pow(dphi,2) );
      }*/
      if (tsos_pixel.isValid()) {
        pixeltrk_pt = pixeltrk->pt();
	pixeltrk_pterr = pixeltrk->ptError();
        pixeltrk_eta = pixeltrk->eta();
        pixeltrk_phi = pixeltrk->phi();
	pixeltrk_dz = pixeltrk->dz(vertices.at(0).position());
	pixeltrk_dxy = pixeltrk->dxy(vertices.at(0).position());
	pixeltrk_validhits = pixeltrk->numberOfValidHits();
	pixeltrk_chi2 = pixeltrk->normalizedChi2();
	pixeltrk_d0sigma = sqrt(pixeltrk->d0Error() * pixeltrk->d0Error() + vertices.at(0).xError() * vertices.at(0).yError());
	pixeltrk_dzsigma = sqrt(pixeltrk->dzError() * pixeltrk->dzError() + vertices.at(0).zError() * vertices.at(0).zError());
	pixeltrk_qoverp = pixeltrk->qoverp();
	pixeltrk_qoverperror = pixeltrk->qoverpError();
      }
      //regression->Fill();
      //continue;
      const reco::Track* hlttrk = NULL;
      for ( const auto & trk : hlttracks) {
	      vrtx_xy = trk.dxy(vertices.at(0).position());
	      vrtx_z = trk.dz(vertices.at(0).position());
	      for (auto const& hit : trk.recHits()) {
                if (!hit->isValid()) continue;
                if (hit->geographicalId() != detId) continue;
		reco::TransientTrack tkTT = theTTrackBuilder->build(trk);
                TrajectoryStateOnSurface tsos = thePropagator->propagate(tkTT.innermostMeasurementState(), geomDet->surface());
                if (!tsos.isValid()) continue;
                auto localValues = stripCPE->localParameters(stripCluster, *stripDet, tsos);
                LocalPoint clusterLocal = localValues.first;

                LocalPoint trackLocal = geomDet->surface().toLocal(tsos.globalPosition());
                float dr = abs(trackLocal.x() - clusterLocal.x()); //std::sqrt( pow( (trackLocal.x() - clusterLocal.x()), 2 ) +
                if (dr < hlttrk_dr_min) {
                  hlttrk_dr_min = dr;
		  hlttrk = &trk;
                }
	      }
      }

      if (hlttrk) {
        hlttrk_pt = hlttrk->pt();
        hlttrk_pterr = hlttrk->ptError();
        hlttrk_eta = hlttrk->eta();
        hlttrk_phi = hlttrk->phi();
        hlttrk_dz = hlttrk->dz(vertices.at(0).position());
        hlttrk_dxy = hlttrk->dxy(vertices.at(0).position());
        hlttrk_validhits = hlttrk->numberOfValidHits();
        hlttrk_chi2 = hlttrk->normalizedChi2();
        hlttrk_d0sigma = sqrt(hlttrk->d0Error() * hlttrk->d0Error() + vertices.at(0).xError() * vertices.at(0).yError());
        hlttrk_dzsigma = sqrt(hlttrk->dzError() * hlttrk->dzError() + vertices.at(0).zError() * vertices.at(0).zError());
        hlttrk_qoverp = hlttrk->qoverp();
        hlttrk_qoverperror = hlttrk->qoverpError();
      }

      tree->Fill();
      if (target) {
	     //std::cout << "dr_pixel:hlttrk_dr_min:recotrk_dr_min " << pixeltrk_dr_min << ":" << hlttrk_dr_min << ":" << recotrk_dr_min << std::endl;
	     /*if (tsos_pixel.isValid() && tsos_reco.isValid()) {
	      std::cout << "eta: " << tsos_pixel.globalPosition().eta() << "\t" << tsos_reco.globalPosition().eta() << std::endl;
	      std::cout << "phi: " << tsos_pixel.globalPosition().phi() << "\t" << tsos_reco.globalPosition().phi() << std::endl;
	     }*/
	      for (unsigned int i = 0; i < adcs.size()-1; ++i) hist_sig->Fill(i, adcs[i]); // Fill 2D histogram
      }
      else {
	      /*if (tsos_pixel.isValid() && tsos_reco.isValid()) {
		  std::cout << "bkg eta: " << tsos_pixel.globalPosition().eta() << "\t" << tsos_reco.globalPosition().eta() << std::endl;
		  std::cout << "phi: " << tsos_pixel.globalPosition().phi() << "\t" << tsos_reco.globalPosition().phi() << std::endl;
						               }
//	      std::cout << "dr_pixel:hlttrk_dr_min:recotrk_dr_min " << pixeltrk_dr_min << ":" << hlttrk_dr_min << ":" << recotrk_dr_min << std::endl;*/
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
  desc.add<edm::InputTag>("hltPixeltracks", edm::InputTag("hltPixelTracks","","HLTX"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("vertex", edm::InputTag("vertex"));
  descriptions.add("nn_tupleProducer_raw", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(nn_tupleProducer_raw);
