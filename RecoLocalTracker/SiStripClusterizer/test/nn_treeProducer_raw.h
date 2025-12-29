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
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

#include "assert.h"
#include "TTree.h"
#include "TH2F.h"
#include "cluster_property.h"

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;

class nn_tupleProducer_raw : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit nn_tupleProducer_raw(const edm::ParameterSet&);
  ~nn_tupleProducer_raw() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void create_tree();
  void initialize_vars();

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

  TTree* tree;
  TTree* regression;
  edm::Service<TFileService> fs;

  edm::EventNumber_t eventN;
  int runN;
  int lumi;

  edm::InputTag beamSpot_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  edm::FileInPath fileInPath_;
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
  float       noise_diff_adc_mone;
  float       noise_diff_adc_mtwo;
  float       noise_diff_adc_mthree;
  float       noise_diff_adc_pone;
  float       noise_diff_adc_ptwo;
  float       noise_diff_adc_pthree;
  int         n_saturated;

  float       ref_hitX[nMax];
  float       ref_hitY[nMax];
  uint16_t    ref_channel[nMax];
  uint16_t    ref_adc[nMax];
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

void nn_tupleProducer_raw::create_tree() {

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
  tree->Branch("noise_max_adc", &noise_max_adc, "noise_max_adc/F");
  tree->Branch("noise_adc_pone", &noise_adc_pone, "noise_adc_pone/F");
  tree->Branch("noise_adc_ptwo", &noise_adc_ptwo, "noise_adc_ptwo/F");
  tree->Branch("noise_adc_pthree", &noise_adc_pthree, "noise_adc_pthree/F");
  tree->Branch("noise_adc_mone", &noise_adc_mone, "noise_adc_mone/F");
  tree->Branch("noise_adc_mtwo", &noise_adc_mtwo, "noise_adc_mtwo/F");
  tree->Branch("noise_adc_mthree", &noise_adc_mthree, "noise_adc_mthree/F");
  tree->Branch("noise_diff_adc_pone", &noise_diff_adc_pone, "noise_diff_adc_pone/F");
  tree->Branch("noise_diff_adc_ptwo", &noise_diff_adc_ptwo, "noise_diff_adc_ptwo/F");
  tree->Branch("noise_diff_adc_pthree", &noise_diff_adc_pthree, "noise_diff_adc_pthree/F");
  tree->Branch("noise_diff_adc_mone", &noise_diff_adc_mone, "noise_diff_adc_mone/F");
  tree->Branch("noise_diff_adc_mtwo", &noise_diff_adc_mtwo, "noise_diff_adc_mtwo/F");
  tree->Branch("noise_diff_adc_mthree", &noise_diff_adc_mthree, "noise_diff_adc_mthree/F");
  tree->Branch("adc_std", &adc_std, "adc_std/F");
}

void nn_tupleProducer_raw::initialize_vars() {

   target = 0;
   vrtx_xy = vrtx_z = 99;

   max_adc_idx = 0;
   max_adc_x = max_adc_y = max_adc_z = 0.;
   max_adc = adc_mone = adc_mtwo = adc_mthree = adc_mfour = adc_pone = adc_ptwo = adc_pthree = adc_pfour = 0;
   diff_adc_mone = diff_adc_mtwo = diff_adc_mthree = diff_adc_pone = diff_adc_ptwo = diff_adc_pthree = 0;
   n_saturated = 0;
   noise_adc_mone = noise_adc_mtwo = noise_adc_mthree = noise_adc_pone = noise_adc_ptwo = noise_adc_pthree = 0;
   noise_diff_adc_mone = noise_diff_adc_mtwo = noise_diff_adc_mthree = noise_diff_adc_pone = noise_diff_adc_ptwo = noise_diff_adc_pthree = 0;

   pixeltrk_pt = 0;
   pixeltrk_eta = pixeltrk_phi = 5;
   pixeltrk_qoverperror = 2;
   pixeltrk_validhits = pixeltrk_qoverp = 0;
   pixeltrk_pterr = 2;
   pixeltrk_chi2 = 1000;
   pixeltrk_dz = pixeltrk_dzsigma = 50;
   pixeltrk_dxy = 2;
   pixeltrk_d0sigma = 50;

   hlttrk_pt = 0;
   hlttrk_eta = hlttrk_phi = 5;
   hlttrk_qoverperror = 2;
   hlttrk_validhits = hlttrk_qoverp = 0;
   hlttrk_pterr = 2;
   hlttrk_chi2 = 1000;
   hlttrk_dz = hlttrk_dzsigma = 50;
   hlttrk_dxy = 2;
   hlttrk_d0sigma = 50;

   recotrk_pt = 0;
   recotrk_eta = recotrk_phi = 5;
   recotrk_qoverperror = 2;
   recotrk_validhits = recotrk_qoverp = 0;
   recotrk_pterr = 2;
   recotrk_chi2 = 1000;
   recotrk_dz = recotrk_dzsigma = 50;
   recotrk_dxy = 2;
   recotrk_d0sigma = 50;
}
