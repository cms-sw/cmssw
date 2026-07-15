/*! \brief   Checklist
 *  \details TTClusters and TTStubs
 *
 *  \author Nicola Pozzobon
 *  \author Sebastien Viret
 *  \date   July 2013
 *  May 2026: Updated by Ian Tomalin to run in CMSSW 15.
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "SimDataFormats/Associations/interface/TTClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/TTStubAssociationMap.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <TH1D.h>
#include <TH2D.h>

class AnalyzerClusterStub : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  /// Public methods
public:
  /// Constructor/destructor
  explicit AnalyzerClusterStub(const edm::ParameterSet& iConfig);
  ~AnalyzerClusterStub() override;
  // Typical methods used on Loops over events
  void beginJob() override;
  void endJob() override;
  void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
  void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  /// Private methods and variables
private:
  /// TrackingParticle and TrackingVertex
  TH2D* hSimVtx_XY;
  TH1D* hSimVtx_Z;

  TH1D* hTPart_Pt;
  TH1D* hTPart_Eta_Pt10;
  TH1D* hTPart_Phi_Pt10;

  /// Global positions of TTClusters
  TH2D* hCluster_Barrel_XY;
  TH2D* hCluster_Barrel_XY_Zoom;
  TH2D* hCluster_Endcap_Fw_XY;
  TH2D* hCluster_Endcap_Bw_XY;
  TH2D* hCluster_RZ;
  TH2D* hCluster_Endcap_Fw_RZ_Zoom;
  TH2D* hCluster_Endcap_Bw_RZ_Zoom;

  TH1D* hCluster_IMem_Barrel;
  TH1D* hCluster_IMem_Endcap;
  TH1D* hCluster_OMem_Barrel;
  TH1D* hCluster_OMem_Endcap;

  TH1D* hCluster_Gen_Barrel;
  TH1D* hCluster_Unkn_Barrel;
  TH1D* hCluster_Comb_Barrel;
  TH1D* hCluster_Gen_Endcap;
  TH1D* hCluster_Unkn_Endcap;
  TH1D* hCluster_Comb_Endcap;

  TH1D* hCluster_Gen_Eta;
  TH1D* hCluster_Unkn_Eta;
  TH1D* hCluster_Comb_Eta;

  TH2D* hCluster_PID;
  TH2D* hCluster_WidthPhi;
  TH2D* hCluster_WidthRZ;

  TH1D* hTPart_Eta_INormalization;
  TH1D* hTPart_Eta_ICW_1;
  TH1D* hTPart_Eta_ICW_2;
  TH1D* hTPart_Eta_ICW_3;
  TH1D* hTPart_Eta_ONormalization;
  TH1D* hTPart_Eta_OCW_1;
  TH1D* hTPart_Eta_OCW_2;
  TH1D* hTPart_Eta_OCW_3;

  /// Global positions of TTStubs
  TH2D* hStub_Barrel_XY;
  TH2D* hStub_Barrel_XY_Zoom;
  TH2D* hStub_Endcap_Fw_XY;
  TH2D* hStub_Endcap_Bw_XY;
  TH2D* hStub_RZ;
  TH2D* hStub_Endcap_Fw_RZ_Zoom;
  TH2D* hStub_Endcap_Bw_RZ_Zoom;

  TH1D* hStub_Barrel;
  TH1D* hStub_Endcap;

  TH1D* hStub_Gen_Barrel;
  TH1D* hStub_Unkn_Barrel;
  TH1D* hStub_Comb_Barrel;
  TH1D* hStub_Gen_Endcap;
  TH1D* hStub_Unkn_Endcap;
  TH1D* hStub_Comb_Endcap;

  TH1D* hStub_Gen_Eta;
  TH1D* hStub_Unkn_Eta;
  TH1D* hStub_Comb_Eta;

  TH1D* hStub_PID;
  TH2D* hStub_Barrel_W;
  TH2D* hStub_Barrel_O;
  TH2D* hStub_Endcap_W;
  TH2D* hStub_Endcap_O;

  /// Stub finding coverage
  TH1D* hTPart_Eta_Pt10_Normalization;
  TH1D* hTPart_Eta_Pt10_NumPS;
  TH1D* hTPart_Eta_Pt10_Num2S;

  /// Denominator for Stub Prod Eff
  std::map<unsigned int, TH1D*> mapCluLayer_hTPart_Pt;
  std::map<unsigned int, TH1D*> mapCluLayer_hTPart_Eta_Pt10;
  std::map<unsigned int, TH1D*> mapCluLayer_hTPart_Phi_Pt10;
  std::map<unsigned int, TH1D*> mapCluDisk_hTPart_Pt;
  std::map<unsigned int, TH1D*> mapCluDisk_hTPart_Eta_Pt10;
  std::map<unsigned int, TH1D*> mapCluDisk_hTPart_Phi_Pt10;
  /// Numerator for Stub Prod Eff
  std::map<unsigned int, TH1D*> mapStubLayer_hTPart_Pt;
  std::map<unsigned int, TH1D*> mapStubLayer_hTPart_Eta_Pt10;
  std::map<unsigned int, TH1D*> mapStubLayer_hTPart_Phi_Pt10;
  std::map<unsigned int, TH1D*> mapStubDisk_hTPart_Pt;
  std::map<unsigned int, TH1D*> mapStubDisk_hTPart_Eta_Pt10;
  std::map<unsigned int, TH1D*> mapStubDisk_hTPart_Phi_Pt10;

  /// Comparison of Stubs to TrackingParticles
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_InvPt_TPart_InvPt;
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_Pt_TPart_Pt;
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_Eta_TPart_Eta;
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_Phi_TPart_Phi;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_InvPt_TPart_InvPt;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_Pt_TPart_Pt;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_Eta_TPart_Eta;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_Phi_TPart_Phi;

  /// Residuals
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_InvPtRes_TPart_Eta;
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_PtRes_TPart_Eta;
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_EtaRes_TPart_Eta;
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_PhiRes_TPart_Eta;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_InvPtRes_TPart_Eta;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_PtRes_TPart_Eta;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_EtaRes_TPart_Eta;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_PhiRes_TPart_Eta;

  /// Stub Width vs Pt
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_W_TPart_Pt;
  std::map<unsigned int, TH2D*> mapStubLayer_hStub_W_TPart_InvPt;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_W_TPart_Pt;
  std::map<unsigned int, TH2D*> mapStubDisk_hStub_W_TPart_InvPt;

  /// Containers of parameters passed by python
  /// configuration file
  edm::ParameterSet config_;

  bool testedGeometry_;
  bool debugMode_;
  double magneticFieldStrength_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> esGetTokenBfield_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> esGetTokenTGeom_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> esGetTokenTTopo_;

  edm::EDGetTokenT<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> ttClusterToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> ttStubToken_;
  edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> ttClusterAssocToken_;
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> ttStubAssocToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> trackingParticleToken_;
  edm::EDGetTokenT<std::vector<TrackingVertex>> trackingVertexToken_;
};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
AnalyzerClusterStub::AnalyzerClusterStub(edm::ParameterSet const& iConfig)
    : config_(iConfig),
      magneticFieldStrength_(0.),
      esGetTokenBfield_(esConsumes<edm::Transition::BeginRun>()),
      esGetTokenTGeom_(esConsumes()),
      esGetTokenTTopo_(esConsumes()) {
  usesResource("TFileService");
  debugMode_ = iConfig.getParameter<bool>("DebugMode");

  edm::InputTag ttClusterInputTag = iConfig.getParameter<edm::InputTag>("TTClusterInputTag");
  edm::InputTag ttStubInputTag = iConfig.getParameter<edm::InputTag>("TTStubInputTag");
  edm::InputTag ttClusterAssocInputTag = iConfig.getParameter<edm::InputTag>("TTClusterAssocInputTag");
  edm::InputTag ttStubAssocInputTag = iConfig.getParameter<edm::InputTag>("TTStubAssocInputTag");
  edm::InputTag trackingParticleInputTag = iConfig.getParameter<edm::InputTag>("TrackingParticleInputTag");
  edm::InputTag trackingVertexInputTag = iConfig.getParameter<edm::InputTag>("TrackingVertexInputTag");

  ttClusterToken_ = consumes<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>(ttClusterInputTag);
  ttStubToken_ = consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(ttStubInputTag);
  ttClusterAssocToken_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>(ttClusterAssocInputTag);
  ttStubAssocToken_ = consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>(ttStubAssocInputTag);
  trackingParticleToken_ = consumes<std::vector<TrackingParticle>>(trackingParticleInputTag);
  trackingVertexToken_ = consumes<std::vector<TrackingVertex>>(trackingVertexInputTag);
}

/////////////
// DESTRUCTOR
AnalyzerClusterStub::~AnalyzerClusterStub() {
  /// Insert here what you need to delete
  /// when you close the class instance
}

//////////
// END JOB
void AnalyzerClusterStub::endJob()  //edm::Run& run, const edm::EventSetup& iSetup
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " AnalyzerClusterStub::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop
}

////////////
// BEGIN RUN
void AnalyzerClusterStub::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  /// Get CMS Bfield
  const MagneticField* theMagneticField = &iSetup.getData(esGetTokenBfield_);
  magneticFieldStrength_ = theMagneticField->inTesla(GlobalPoint(0, 0, 0)).z();
}

////////////
// BEGIN JOB
void AnalyzerClusterStub::beginJob() {
  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  testedGeometry_ = false;

  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " AnalyzerClusterStub::beginJob" << std::endl;

  /// Book histograms etc
  edm::Service<TFileService> fs;

  /// Prepare for LogXY Plots
  int NumBins = 200;
  double MinPt = 0.0;
  double MaxPt = 100.0;

  double* BinVec = new double[NumBins + 1];
  for (int iBin = 0; iBin < NumBins + 1; iBin++) {
    double temp = pow(10, (-NumBins + iBin) / (MaxPt - MinPt));
    BinVec[iBin] = temp;
  }

  /// TrackingParticle and TrackingVertex
  hSimVtx_XY = fs->make<TH2D>("hSimVtx_XY", "SimVtx y vs. x", 40, -0.4, 0.4, 40, -0.4, 0.4);
  hSimVtx_Z = fs->make<TH1D>("hSimVtx_Z", "SimVtx z", 100, -25, 25);
  hSimVtx_XY->Sumw2();
  hSimVtx_Z->Sumw2();

  hTPart_Pt = fs->make<TH1D>("hTPart_Pt", "TPart p_{T}", 200, 0, 50);
  hTPart_Eta_Pt10 = fs->make<TH1D>("hTPart_Eta_Pt10", "TPart #eta (p_{T} > 10 GeV/c)", 180, -M_PI, M_PI);
  hTPart_Phi_Pt10 = fs->make<TH1D>("hTPart_Phi_Pt10", "TPart #phi (p_{T} > 10 GeV/c)", 180, -M_PI, M_PI);
  hTPart_Pt->Sumw2();
  hTPart_Eta_Pt10->Sumw2();
  hTPart_Phi_Pt10->Sumw2();

  /// Global position of TTCluster
  hCluster_Barrel_XY = fs->make<TH2D>("hCluster_Barrel_XY", "TTCluster Barrel y vs. x", 960, -120, 120, 960, -120, 120);
  hCluster_Barrel_XY_Zoom =
      fs->make<TH2D>("hCluster_Barrel_XY_Zoom", "TTCluster Barrel y vs. x", 960, 30, 60, 960, -15, 15);
  hCluster_Endcap_Fw_XY =
      fs->make<TH2D>("hCluster_Endcap_Fw_XY", "TTCluster Forward Endcap y vs. x", 960, -120, 120, 960, -120, 120);
  hCluster_Endcap_Bw_XY =
      fs->make<TH2D>("hCluster_Endcap_Bw_XY", "TTCluster Backward Endcap y vs. x", 960, -120, 120, 960, -120, 120);
  hCluster_RZ = fs->make<TH2D>("hCluster_RZ", "TTCluster #rho vs. z", 900, -300, 300, 480, 0, 120);
  hCluster_Endcap_Fw_RZ_Zoom =
      fs->make<TH2D>("hCluster_Endcap_Fw_RZ_Zoom", "TTCluster Forward Endcap #rho vs. z", 960, 140, 170, 960, 30, 60);
  hCluster_Endcap_Bw_RZ_Zoom = fs->make<TH2D>(
      "hCluster_Endcap_Bw_RZ_Zoom", "TTCluster Backward Endcap #rho vs. z", 960, -170, -140, 960, 70, 100);
  hCluster_Barrel_XY->Sumw2();
  hCluster_Barrel_XY_Zoom->Sumw2();
  hCluster_Endcap_Fw_XY->Sumw2();
  hCluster_Endcap_Bw_XY->Sumw2();
  hCluster_RZ->Sumw2();
  hCluster_Endcap_Fw_RZ_Zoom->Sumw2();
  hCluster_Endcap_Bw_RZ_Zoom->Sumw2();

  hCluster_IMem_Barrel = fs->make<TH1D>("hCluster_IMem_Barrel", "Inner TTCluster Layer", 12, -0.5, 11.5);
  hCluster_IMem_Endcap = fs->make<TH1D>("hCluster_IMem_Endcap", "Inner TTCluster Disk", 12, -0.5, 11.5);
  hCluster_OMem_Barrel = fs->make<TH1D>("hCluster_OMem_Barrel", "Outer TTCluster Layer", 12, -0.5, 11.5);
  hCluster_OMem_Endcap = fs->make<TH1D>("hCluster_OMem_Endcap", "Outer TTCluster Disk", 12, -0.5, 11.5);
  hCluster_IMem_Barrel->Sumw2();
  hCluster_IMem_Endcap->Sumw2();
  hCluster_OMem_Barrel->Sumw2();
  hCluster_OMem_Endcap->Sumw2();

  hCluster_Gen_Barrel = fs->make<TH1D>("hCluster_Gen_Barrel", "Genuine TTCluster Layer", 12, -0.5, 11.5);
  hCluster_Unkn_Barrel = fs->make<TH1D>("hCluster_Unkn_Barrel", "Unknown TTCluster Layer", 12, -0.5, 11.5);
  hCluster_Comb_Barrel = fs->make<TH1D>("hCluster_Comb_Barrel", "Combinatorial TTCluster Layer", 12, -0.5, 11.5);
  hCluster_Gen_Endcap = fs->make<TH1D>("hCluster_Gen_Endcap", "Genuine TTCluster Disk", 12, -0.5, 11.5);
  hCluster_Unkn_Endcap = fs->make<TH1D>("hCluster_Unkn_Endcap", "Unknown TTCluster Disk", 12, -0.5, 11.5);
  hCluster_Comb_Endcap = fs->make<TH1D>("hCluster_Comb_Endcap", "Combinatorial TTCluster Disk", 12, -0.5, 11.5);
  hCluster_Gen_Barrel->Sumw2();
  hCluster_Unkn_Barrel->Sumw2();
  hCluster_Comb_Barrel->Sumw2();
  hCluster_Gen_Endcap->Sumw2();
  hCluster_Unkn_Endcap->Sumw2();
  hCluster_Comb_Endcap->Sumw2();

  hCluster_Gen_Eta = fs->make<TH1D>("hCluster_Gen_Eta", "Genuine TTCluster #eta", 90, 0, M_PI);
  hCluster_Unkn_Eta = fs->make<TH1D>("hCluster_Unkn_Eta", "Unknown TTCluster #eta", 90, 0, M_PI);
  hCluster_Comb_Eta = fs->make<TH1D>("hCluster_Comb_Eta", "Combinatorial TTCluster #eta", 90, 0, M_PI);
  hCluster_Gen_Eta->Sumw2();
  hCluster_Unkn_Eta->Sumw2();
  hCluster_Comb_Eta->Sumw2();

  hCluster_PID = fs->make<TH2D>("hCluster_PID", "Stack member vs TTCluster PID", 501, -250.5, 250.5, 2, -0.5, 1.5);
  hCluster_WidthPhi =
      fs->make<TH2D>("hCluster_WidthPhi", "Stack member vs TTCluster Width (pitch units)", 10, -0.5, 9.5, 2, -0.5, 1.5);
  hCluster_WidthRZ =
      fs->make<TH2D>("hCluster_WidthRZ", "Stack member vs TTCluster Width (pitch units)", 10, -0.5, 9.5, 2, -0.5, 1.5);
  hCluster_PID->Sumw2();
  hCluster_WidthPhi->Sumw2();
  hCluster_WidthRZ->Sumw2();

  hTPart_Eta_INormalization = fs->make<TH1D>("hTPart_Eta_INormalization", "TParticles vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_ICW_1 = fs->make<TH1D>("hTPart_Eta_ICW_1", "CW 1 vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_ICW_2 = fs->make<TH1D>("hTPart_Eta_ICW_2", "CW 2 vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_ICW_3 = fs->make<TH1D>("hTPart_Eta_ICW_3", "CW 3 or more vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_INormalization->Sumw2();
  hTPart_Eta_ICW_1->Sumw2();
  hTPart_Eta_ICW_2->Sumw2();
  hTPart_Eta_ICW_3->Sumw2();

  hTPart_Eta_ONormalization = fs->make<TH1D>("hTPart_Eta_ONormalization", "TParticles vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_OCW_1 = fs->make<TH1D>("hTPart_Eta_OCW_1", "CW 1 vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_OCW_2 = fs->make<TH1D>("hTPart_Eta_OCW_2", "CW 2 vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_OCW_3 = fs->make<TH1D>("hTPart_Eta_OCW_3", "CW 3 or more vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_ONormalization->Sumw2();
  hTPart_Eta_OCW_1->Sumw2();
  hTPart_Eta_OCW_2->Sumw2();
  hTPart_Eta_OCW_3->Sumw2();

  /// Global position of TTStub
  hStub_Barrel_XY = fs->make<TH2D>("hStub_Barrel_XY", "TTStub Barrel y vs. x", 960, -120, 120, 960, -120, 120);
  hStub_Barrel_XY_Zoom = fs->make<TH2D>("hStub_Barrel_XY_Zoom", "TTStub Barrel y vs. x", 960, 30, 60, 960, -15, 15);
  hStub_Endcap_Fw_XY =
      fs->make<TH2D>("hStub_Endcap_Fw_XY", "TTStub Forward Endcap y vs. x", 960, -120, 120, 960, -120, 120);
  hStub_Endcap_Bw_XY =
      fs->make<TH2D>("hStub_Endcap_Bw_XY", "TTStub Backward Endcap y vs. x", 960, -120, 120, 960, -120, 120);
  hStub_RZ = fs->make<TH2D>("hStub_RZ", "TTStub #rho vs. z", 900, -300, 300, 480, 0, 120);
  hStub_Endcap_Fw_RZ_Zoom =
      fs->make<TH2D>("hStub_Endcap_Fw_RZ_Zoom", "TTStub Forward Endcap #rho vs. z", 960, 140, 170, 960, 30, 100);
  hStub_Endcap_Bw_RZ_Zoom =
      fs->make<TH2D>("hStub_Endcap_Bw_RZ_Zoom", "TTStub Backward Endcap #rho vs. z", 960, -170, -140, 960, 30, 100);
  hStub_Barrel_XY->Sumw2();
  hStub_Barrel_XY_Zoom->Sumw2();
  hStub_Endcap_Fw_XY->Sumw2();
  hStub_Endcap_Bw_XY->Sumw2();
  hStub_RZ->Sumw2();
  hStub_Endcap_Fw_RZ_Zoom->Sumw2();
  hStub_Endcap_Bw_RZ_Zoom->Sumw2();

  hStub_Barrel = fs->make<TH1D>("hStub_Barrel", "TTStub Layer", 12, -0.5, 11.5);
  hStub_Endcap = fs->make<TH1D>("hStub_Endcap", "TTStub Disk", 12, -0.5, 11.5);
  hStub_Barrel->Sumw2();
  hStub_Endcap->Sumw2();

  hStub_Gen_Barrel = fs->make<TH1D>("hStub_Gen_Barrel", "Genuine TTStub Layer", 12, -0.5, 11.5);
  hStub_Unkn_Barrel = fs->make<TH1D>("hStub_Unkn_Barrel", "Unknown  TTStub Layer", 12, -0.5, 11.5);
  hStub_Comb_Barrel = fs->make<TH1D>("hStub_Comb_Barrel", "Combinatorial TTStub Layer", 12, -0.5, 11.5);
  hStub_Gen_Endcap = fs->make<TH1D>("hStub_Gen_Endcap", "Genuine TTStub Disk", 12, -0.5, 11.5);
  hStub_Unkn_Endcap = fs->make<TH1D>("hStub_Unkn_Endcap", "Unknown  TTStub Disk", 12, -0.5, 11.5);
  hStub_Comb_Endcap = fs->make<TH1D>("hStub_Comb_Endcap", "Combinatorial TTStub Disk", 12, -0.5, 11.5);
  hStub_Gen_Barrel->Sumw2();
  hStub_Unkn_Barrel->Sumw2();
  hStub_Comb_Barrel->Sumw2();
  hStub_Gen_Endcap->Sumw2();
  hStub_Unkn_Endcap->Sumw2();
  hStub_Comb_Endcap->Sumw2();

  hStub_Gen_Eta = fs->make<TH1D>("hStub_Gen_Eta", "Genuine TTStub #eta", 90, 0, M_PI);
  hStub_Unkn_Eta = fs->make<TH1D>("hStub_Unkn_Eta", "Unknown TTStub #eta", 90, 0, M_PI);
  hStub_Comb_Eta = fs->make<TH1D>("hStub_Comb_Eta", "Combinatorial TTStub #eta", 90, 0, M_PI);
  hStub_Gen_Eta->Sumw2();
  hStub_Unkn_Eta->Sumw2();
  hStub_Comb_Eta->Sumw2();

  hStub_PID = fs->make<TH1D>("hStub_PID", "TTStub PID", 501, -250.5, 250.5);
  hStub_Barrel_W =
      fs->make<TH2D>("hStub_Barrel_W", "TTStub Post-Corr Displacement (Layer)", 12, -0.5, 11.5, 43, -10.75, 10.75);
  hStub_Barrel_O = fs->make<TH2D>("hStub_Barrel_O", "TTStub Offset (Layer)", 12, -0.5, 11.5, 43, -10.75, 10.75);
  hStub_Endcap_W =
      fs->make<TH2D>("hStub_Endcap_W", "TTStub Post-Corr Displacement (Layer)", 12, -0.5, 11.5, 43, -10.75, 10.75);
  hStub_Endcap_O = fs->make<TH2D>("hStub_Endcap_O", "TTStub Offset (Layer)", 12, -0.5, 11.5, 43, -10.75, 10.75);

  hStub_PID->Sumw2();
  hStub_Barrel_W->Sumw2();
  hStub_Barrel_O->Sumw2();
  hStub_Endcap_W->Sumw2();
  hStub_Endcap_O->Sumw2();

  hTPart_Eta_Pt10_Normalization =
      fs->make<TH1D>("hTPart_Eta_Pt10_Normalization", "TParticles vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_Pt10_NumPS = fs->make<TH1D>("hTPart_Eta_Pt10_NumPS", "PS Stubs vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_Pt10_Num2S = fs->make<TH1D>("hTPart_Eta_Pt10_Num2S", "2S Stubs vs. TPart #eta", 90, 0, M_PI);
  hTPart_Eta_Pt10_Normalization->Sumw2();
  hTPart_Eta_Pt10_NumPS->Sumw2();
  hTPart_Eta_Pt10_Num2S->Sumw2();

  /// Stub Production Efficiency and comparison to TrackingParticle per layer/disk

  constexpr unsigned int numLayers = 6;
  for (unsigned int lay = 1; lay <= numLayers; lay++) {
    /// BARREL

    /// Denominators
    histoName.str("");
    histoName << "hTPart_Pt_Clu_L" << lay;
    histoTitle.str("");
    histoTitle << "TPart p_{T}, Cluster, Barrel Layer " << lay;
    mapCluLayer_hTPart_Pt[lay] = fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0, 50);
    histoName.str("");
    histoName << "hTPart_Eta_Pt10_Clu_L" << lay;
    histoTitle.str("");
    histoTitle << "TPart #eta (p_{T} > 10 GeV/c), Cluster, Barrel Layer " << lay;
    mapCluLayer_hTPart_Eta_Pt10[lay] =
        fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI);
    histoName.str("");
    histoName << "hTPart_Phi_Pt10_Clu_L" << lay;
    histoTitle.str("");
    histoTitle << "TPart #phi (p_{T} > 10 GeV/c), Cluster, Barrel Layer " << lay;
    mapCluLayer_hTPart_Phi_Pt10[lay] =
        fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI);
    mapCluLayer_hTPart_Pt[lay]->Sumw2();
    mapCluLayer_hTPart_Eta_Pt10[lay]->Sumw2();
    mapCluLayer_hTPart_Phi_Pt10[lay]->Sumw2();

    /// Numerators GeV/c
    histoName.str("");
    histoName << "hTPart_Pt_Stub_L" << lay;
    histoTitle.str("");
    histoTitle << "TPart p_{T}, Stub, Barrel Layer " << lay;
    mapStubLayer_hTPart_Pt[lay] = fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0, 50);
    histoName.str("");
    histoName << "hTPart_Eta_Pt10_Stub_L" << lay;
    histoTitle.str("");
    histoTitle << "TPart #eta (p_{T} > 10 GeV/c), Stub, Barrel Layer " << lay;
    mapStubLayer_hTPart_Eta_Pt10[lay] =
        fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI);
    histoName.str("");
    histoName << "hTPart_Phi_Pt10_Stub_L" << lay;
    histoTitle.str("");
    histoTitle << "TPart #phi (p_{T} > 10 GeV/c), Stub, Barrel Layer " << lay;
    mapStubLayer_hTPart_Phi_Pt10[lay] =
        fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI);
    mapStubLayer_hTPart_Pt[lay]->Sumw2();
    mapStubLayer_hTPart_Eta_Pt10[lay]->Sumw2();
    mapStubLayer_hTPart_Phi_Pt10[lay]->Sumw2();

    /// Comparison to TrackingParticle
    histoName.str("");
    histoName << "hStub_InvPt_TPart_InvPt_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub p_{T}^{-1} vs. TPart p_{T}^{-1}, Barrel Layer " << lay;
    mapStubLayer_hStub_InvPt_TPart_InvPt[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0.0, 0.8, 200, 0.0, 0.8);
    mapStubLayer_hStub_InvPt_TPart_InvPt[lay]->GetXaxis()->Set(NumBins, BinVec);
    mapStubLayer_hStub_InvPt_TPart_InvPt[lay]->GetYaxis()->Set(NumBins, BinVec);
    mapStubLayer_hStub_InvPt_TPart_InvPt[lay]->Sumw2();

    histoName.str("");
    histoName << "hStub_Pt_TPart_Pt_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub p_{T} vs. TPart p_{T}, Barrel Layer " << lay;
    mapStubLayer_hStub_Pt_TPart_Pt[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 100, 0, 50, 100, 0, 50);
    mapStubLayer_hStub_Pt_TPart_Pt[lay]->Sumw2();

    histoName.str("");
    histoName << "hStub_Eta_TPart_Eta_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub #eta vs. TPart #eta, Barrel Layer " << lay;
    mapStubLayer_hStub_Eta_TPart_Eta[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 180, -M_PI, M_PI);
    mapStubLayer_hStub_Eta_TPart_Eta[lay]->Sumw2();

    histoName.str("");
    histoName << "hStub_Phi_TPart_Phi_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub #phi vs. TPart #phi, Barrel Layer " << lay;
    mapStubLayer_hStub_Phi_TPart_Phi[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 180, -M_PI, M_PI);
    mapStubLayer_hStub_Phi_TPart_Phi[lay]->Sumw2();

    /// Residuals
    histoName.str("");
    histoName << "hStub_InvPtRes_TPart_Eta_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub p_{T}^{-1} - TPart p_{T}^{-1} vs. TPart #eta, Barrel Layer " << lay;
    mapStubLayer_hStub_InvPtRes_TPart_Eta[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 100, -2.0, 2.0);
    mapStubLayer_hStub_InvPtRes_TPart_Eta[lay]->Sumw2();

    histoName.str("");
    histoName << "hStub_PtRes_TPart_Eta_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub p_{T} - TPart p_{T} vs. TPart #eta, Barrel Layer " << lay;
    mapStubLayer_hStub_PtRes_TPart_Eta[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 100, -40, 40);
    mapStubLayer_hStub_PtRes_TPart_Eta[lay]->Sumw2();

    histoName.str("");
    histoName << "hStub_EtaRes_TPart_Eta_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub #eta - TPart #eta vs. TPart #eta, Barrel Layer " << lay;
    mapStubLayer_hStub_EtaRes_TPart_Eta[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 100, -2, 2);
    mapStubLayer_hStub_EtaRes_TPart_Eta[lay]->Sumw2();

    histoName.str("");
    histoName << "hStub_PhiRes_TPart_Eta_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub #phi - TPart #phi vs. TPart #eta, Barrel Layer " << lay;
    mapStubLayer_hStub_PhiRes_TPart_Eta[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 100, -0.5, 0.5);
    mapStubLayer_hStub_PhiRes_TPart_Eta[lay]->Sumw2();

    /// Stub Width vs. Pt
    histoName.str("");
    histoName << "hStub_W_TPart_Pt_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub Width vs. TPart p_{T}, Barrel Layer " << lay;
    mapStubLayer_hStub_W_TPart_Pt[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0, 50, 41, -10.25, 10.25);
    mapStubLayer_hStub_W_TPart_Pt[lay]->Sumw2();

    histoName.str("");
    histoName << "hStub_W_TPart_InvPt_L" << lay;
    histoTitle.str("");
    histoTitle << "Stub Width vs. TPart p_{T}^{-1}, Barrel Layer " << lay;
    mapStubLayer_hStub_W_TPart_InvPt[lay] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0, 0.8, 41, -10.25, 10.25);
    mapStubLayer_hStub_W_TPart_InvPt[lay]->GetXaxis()->Set(NumBins, BinVec);
    mapStubLayer_hStub_W_TPart_InvPt[lay]->Sumw2();
  }

  constexpr unsigned int numDisks = 5;
  for (unsigned int disk = 1; disk <= numDisks; disk++) {
    /// ENDCAP

    /// Denominators
    histoName.str("");
    histoName << "hTPart_Pt_Clu_D" << disk;
    histoTitle.str("");
    histoTitle << "TPart p_{T}, Cluster, Endcap Disk " << disk;
    mapCluDisk_hTPart_Pt[disk] = fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0, 50);
    histoName.str("");
    histoName << "hTPart_Eta_Pt10_Clu_D" << disk;
    histoTitle.str("");
    histoTitle << "TPart #eta (p_{T} > 10 GeV/c), Cluster, Endcap Disk " << disk;
    mapCluDisk_hTPart_Eta_Pt10[disk] =
        fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI);
    histoName.str("");
    histoName << "hTPart_Phi_Pt10_Clu_D" << disk;
    histoTitle.str("");
    histoTitle << "TPart #phi (p_{T} > 10 GeV/c), Cluster, Endcap Disk " << disk;
    mapCluDisk_hTPart_Phi_Pt10[disk] =
        fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI);
    mapCluDisk_hTPart_Pt[disk]->Sumw2();
    mapCluDisk_hTPart_Eta_Pt10[disk]->Sumw2();
    mapCluDisk_hTPart_Phi_Pt10[disk]->Sumw2();

    /// Numerators GeV/c
    histoName.str("");
    histoName << "hTPart_Pt_Stub_D" << disk;
    histoTitle.str("");
    histoTitle << "TPart p_{T}, Stub, Endcap Disk " << disk;
    mapStubDisk_hTPart_Pt[disk] = fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0, 50);
    histoName.str("");
    histoName << "hTPart_Eta_Pt10_Stub_D" << disk;
    histoTitle.str("");
    histoTitle << "TPart #eta (p_{T} > 10 GeV/c), Stub, Endcap Disk " << disk;
    mapStubDisk_hTPart_Eta_Pt10[disk] =
        fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI);
    histoName.str("");
    histoName << "hTPart_Phi_Pt10_Stub_D" << disk;
    histoTitle.str("");
    histoTitle << "TPart #phi (p_{T} > 10 GeV/c), Stub, Endcap Disk " << disk;
    mapStubDisk_hTPart_Phi_Pt10[disk] =
        fs->make<TH1D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI);
    mapStubDisk_hTPart_Pt[disk]->Sumw2();
    mapStubDisk_hTPart_Eta_Pt10[disk]->Sumw2();
    mapStubDisk_hTPart_Phi_Pt10[disk]->Sumw2();

    /// Comparison to TrackingParticle
    histoName.str("");
    histoName << "hStub_InvPt_TPart_InvPt_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub p_{T}^{-1} vs. TPart p_{T}^{-1}, Endcap Disk " << disk;
    mapStubDisk_hStub_InvPt_TPart_InvPt[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0.0, 0.8, 200, 0.0, 0.8);
    mapStubDisk_hStub_InvPt_TPart_InvPt[disk]->GetXaxis()->Set(NumBins, BinVec);
    mapStubDisk_hStub_InvPt_TPart_InvPt[disk]->GetYaxis()->Set(NumBins, BinVec);
    mapStubDisk_hStub_InvPt_TPart_InvPt[disk]->Sumw2();

    histoName.str("");
    histoName << "hStub_Pt_TPart_Pt_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub p_{T} vs. TPart p_{T}, Endcap Disk " << disk;
    mapStubDisk_hStub_Pt_TPart_Pt[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 100, 0, 50, 100, 0, 50);
    mapStubDisk_hStub_Pt_TPart_Pt[disk]->Sumw2();

    histoName.str("");
    histoName << "hStub_Eta_TPart_Eta_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub #eta vs. TPart #eta, Endcap Disk " << disk;
    mapStubDisk_hStub_Eta_TPart_Eta[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 180, -M_PI, M_PI);
    mapStubDisk_hStub_Eta_TPart_Eta[disk]->Sumw2();

    histoName.str("");
    histoName << "hStub_Phi_TPart_Phi_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub #phi vs. TPart #phi, Endcap Disk " << disk;
    mapStubDisk_hStub_Phi_TPart_Phi[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 180, -M_PI, M_PI);
    mapStubDisk_hStub_Phi_TPart_Phi[disk]->Sumw2();

    /// Residuals
    histoName.str("");
    histoName << "hStub_InvPtRes_TPart_Eta_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub p_{T}^{-1} - TPart p_{T}^{-1} vs. TPart #eta, Endcap Disk " << disk;
    mapStubDisk_hStub_InvPtRes_TPart_Eta[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 100, -2.0, 2.0);
    mapStubDisk_hStub_InvPtRes_TPart_Eta[disk]->Sumw2();

    histoName.str("");
    histoName << "hStub_PtRes_TPart_Eta_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub p_{T} - TPart p_{T} vs. TPart #eta, Endcap Disk " << disk;
    mapStubDisk_hStub_PtRes_TPart_Eta[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 100, -40, 40);
    mapStubDisk_hStub_PtRes_TPart_Eta[disk]->Sumw2();

    histoName.str("");
    histoName << "hStub_EtaRes_TPart_Eta_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub #eta - TPart #eta vs. TPart #eta, Endcap Disk " << disk;
    mapStubDisk_hStub_EtaRes_TPart_Eta[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 100, -2, 2);
    mapStubDisk_hStub_EtaRes_TPart_Eta[disk]->Sumw2();

    histoName.str("");
    histoName << "hStub_PhiRes_TPart_Eta_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub #phi - TPart #phi vs. TPart #eta, Endcap Disk " << disk;
    mapStubDisk_hStub_PhiRes_TPart_Eta[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 180, -M_PI, M_PI, 100, -0.5, 0.5);
    mapStubDisk_hStub_PhiRes_TPart_Eta[disk]->Sumw2();

    /// Stub Width vs. Pt
    histoName.str("");
    histoName << "hStub_W_TPart_Pt_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub Width vs. TPart p_{T}, Endcap Disk " << disk;
    mapStubDisk_hStub_W_TPart_Pt[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0, 50, 41, -10.25, 10.25);
    mapStubDisk_hStub_W_TPart_Pt[disk]->Sumw2();

    histoName.str("");
    histoName << "hStub_W_TPart_InvPt_D" << disk;
    histoTitle.str("");
    histoTitle << "Stub Width vs. TPart p_{T}^{-1}, Endcap Disk " << disk;
    mapStubDisk_hStub_W_TPart_InvPt[disk] =
        fs->make<TH2D>(histoName.str().c_str(), histoTitle.str().c_str(), 200, 0, 0.8, 41, -10.25, 10.25);
    mapStubDisk_hStub_W_TPart_InvPt[disk]->GetXaxis()->Set(NumBins, BinVec);
    mapStubDisk_hStub_W_TPart_InvPt[disk]->Sumw2();
  }

  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void AnalyzerClusterStub::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get tracker geometry
  const TrackerGeometry* const trackerGeom = &iSetup.getData(esGetTokenTGeom_);
  const TrackerTopology* const trackerTopo = &iSetup.getData(esGetTokenTTopo_);

  // Get Clusters, stubs and truth.
  edm::Handle<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> ttClusterHandle;
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> ttStubHandle;
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> ttClusterAssocHandle;
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> ttStubAssocHandle;
  edm::Handle<std::vector<TrackingParticle>> trackingParticleHandle;
  edm::Handle<std::vector<TrackingVertex>> trackingVertexHandle;

  iEvent.getByToken(ttClusterToken_, ttClusterHandle);
  iEvent.getByToken(ttStubToken_, ttStubHandle);
  iEvent.getByToken(ttClusterAssocToken_, ttClusterAssocHandle);
  iEvent.getByToken(ttStubAssocToken_, ttStubAssocHandle);
  iEvent.getByToken(trackingParticleToken_, trackingParticleHandle);
  iEvent.getByToken(trackingVertexToken_, trackingVertexHandle);

  ////////////////////////////////
  /// COLLECT STUB INFORMATION ///
  ////////////////////////////////

  /// Eta coverage
  /// Go on only if there are TrackingParticles
  if (!trackingParticleHandle->empty()) {
    /// Loop over TrackingParticles
    unsigned int tpCnt = 0;
    std::vector<TrackingParticle>::const_iterator iterTP;
    for (iterTP = trackingParticleHandle->begin(); iterTP != trackingParticleHandle->end(); ++iterTP) {
      /// FIX ADD FILTER TO SELECT INTERESTING TRACKS

      /// Make the pointer
      edm::Ptr<TrackingParticle> tempTPPtr(trackingParticleHandle, tpCnt++);

      /// Search the cluster MC map
      std::vector<edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>>
          theseClusters = ttClusterAssocHandle->findTTClusterRefs(tempTPPtr);

      if (!theseClusters.empty()) {
        bool normIClu = false;
        bool normOClu = false;

        /// Loop over the Clusters
        for (unsigned int jc = 0; jc < theseClusters.size(); jc++) {
          /// Check if it is good
          bool genuineClu = ttClusterAssocHandle->isGenuine(theseClusters.at(jc));
          if (!genuineClu)
            continue;

          unsigned int stackMember = theseClusters.at(jc)->getStackMember();
          unsigned int clusterWidth = theseClusters.at(jc)->findWidth();

          if (stackMember == 0) {
            if (normIClu == false) {
              hTPart_Eta_INormalization->Fill(fabs(tempTPPtr->momentum().eta()));
              normIClu = true;
            }

            if (clusterWidth == 1) {
              hTPart_Eta_ICW_1->Fill(fabs(tempTPPtr->momentum().eta()));
            } else if (clusterWidth == 2) {
              hTPart_Eta_ICW_2->Fill(fabs(tempTPPtr->momentum().eta()));
            } else {
              hTPart_Eta_ICW_3->Fill(fabs(tempTPPtr->momentum().eta()));
            }
          } else if (stackMember == 1) {
            if (normOClu == false) {
              hTPart_Eta_ONormalization->Fill(fabs(tempTPPtr->momentum().eta()));
              normOClu = true;
            }

            if (clusterWidth == 1) {
              hTPart_Eta_OCW_1->Fill(fabs(tempTPPtr->momentum().eta()));
            } else if (clusterWidth == 2) {
              hTPart_Eta_OCW_2->Fill(fabs(tempTPPtr->momentum().eta()));
            } else {
              hTPart_Eta_OCW_3->Fill(fabs(tempTPPtr->momentum().eta()));
            }
          }
        }  /// End of loop over clusters
      }

      /// Search the stub MC truth map
      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
          theseStubs = ttStubAssocHandle->findTTStubRefs(tempTPPtr);

      if (tempTPPtr->p4().pt() <= 10)
        continue;

      if (!theseStubs.empty()) {
        bool normStub = false;

        /// Loop over the Stubs
        for (unsigned int js = 0; js < theseStubs.size(); js++) {
          /// Check if it is good
          bool genuineStub = ttStubAssocHandle->isGenuine(theseStubs.at(js));
          if (!genuineStub)
            continue;

          if (normStub == false) {
            hTPart_Eta_Pt10_Normalization->Fill(fabs(tempTPPtr->momentum().eta()));
            normStub = true;
          }

          /// Classify the stub
          DetId stDetId(theseStubs.at(js)->getDetId());
          /// Check if there are PS modules in seed or candidate
          const GeomDetUnit* det0 = trackerGeom->idToDetUnit(stDetId + 1);
          const GeomDetUnit* det1 = trackerGeom->idToDetUnit(stDetId + 2);

          /// Find pixel pitch and topology related information
          const PixelGeomDetUnit* pix0 = dynamic_cast<const PixelGeomDetUnit*>(det0);
          const PixelGeomDetUnit* pix1 = dynamic_cast<const PixelGeomDetUnit*>(det1);
          const PixelTopology* top0 = dynamic_cast<const PixelTopology*>(&(pix0->specificTopology()));
          const PixelTopology* top1 = dynamic_cast<const PixelTopology*>(&(pix1->specificTopology()));
          int cols0 = top0->ncolumns();
          int cols1 = top1->ncolumns();
          int ratio = cols0 / cols1;  /// This assumes the ratio is integer!

          if (ratio == 1)  /// 2S Modules
          {
            hTPart_Eta_Pt10_Num2S->Fill(fabs(tempTPPtr->momentum().eta()));
          } else  /// PS
          {
            hTPart_Eta_Pt10_NumPS->Fill(fabs(tempTPPtr->momentum().eta()));
          }
        }  /// End of loop over the Stubs generated by this TrackingParticle
      }
    }  /// End of loop over TrackingParticles
  }

  /// Maps to store TrackingParticle information
  std::map<unsigned int, std::vector<edm::Ptr<TrackingParticle>>> tpPerLayer;
  std::map<unsigned int, std::vector<edm::Ptr<TrackingParticle>>> tpPerDisk;

  /// Loop over the input Clusters
  typename edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>::const_iterator inputIter;
  typename edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>>::const_iterator contentIter;
  for (inputIter = ttClusterHandle->begin(); inputIter != ttClusterHandle->end(); ++inputIter) {
    for (contentIter = inputIter->begin(); contentIter != inputIter->end(); ++contentIter) {
      /// Make the reference to be put in the map
      edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>> tempCluRef =
          edmNew::makeRefTo(ttClusterHandle, contentIter);

      DetId detIdClu(tempCluRef->getDetId());
      unsigned int memberClu = tempCluRef->getStackMember();  // upper or lower sensor
      bool genuineClu = ttClusterAssocHandle->isGenuine(tempCluRef);
      bool combinClu = ttClusterAssocHandle->isCombinatoric(tempCluRef);
      //bool unknownClu     = ttClusterAssocHandle->isUnknown( tempCluRef );
      int partClu = 999999999;
      if (genuineClu) {
        edm::Ptr<TrackingParticle> thisTP = ttClusterAssocHandle->findTrackingParticlePtr(tempCluRef);
        partClu = thisTP->pdgId();
      }
      unsigned int widClu = tempCluRef->findWidth();
      unsigned int widCluCols = tempCluRef->findWidthCols();

      // Get cluster position
      const GeomDetUnit* det = trackerGeom->idToDetUnit(detIdClu);
      const PixelTopology* pixTopo =
          dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(det)->specificTopology()));
      const Plane& plane = dynamic_cast<const PixelGeomDetUnit*>(det)->surface();
      const MeasurementPoint& mp = tempCluRef->findAverageLocalCoordinatesCentered();
      GlobalPoint posClu = plane.toGlobal(pixTopo->localPosition(mp));

      hCluster_RZ->Fill(posClu.z(), posClu.perp());

      bool barrel =
          detIdClu.subdetId() == Phase2Tracker::Subdetector::Barrel || detIdClu.subdetId() == StripSubdetector::TIB;
      unsigned int layer = trackerTopo->layer(detIdClu);         // 1-6 in barrel
      unsigned int disk = trackerTopo->endcapWheelP2(detIdClu);  // 1-5 in endcap

      if (barrel) {
        if (memberClu == 0) {
          hCluster_IMem_Barrel->Fill(layer);
        } else {
          hCluster_OMem_Barrel->Fill(layer);
        }

        if (genuineClu) {
          hCluster_Gen_Barrel->Fill(layer);
        } else if (combinClu) {
          hCluster_Comb_Barrel->Fill(layer);
        } else {
          hCluster_Unkn_Barrel->Fill(layer);
        }

        hCluster_Barrel_XY->Fill(posClu.x(), posClu.y());
        hCluster_Barrel_XY_Zoom->Fill(posClu.x(), posClu.y());
      } else {
        // endcap
        if (memberClu == 0) {
          hCluster_IMem_Endcap->Fill(disk);
        } else {
          hCluster_OMem_Endcap->Fill(disk);
        }

        if (genuineClu) {
          hCluster_Gen_Endcap->Fill(disk);
        } else if (combinClu) {
          hCluster_Comb_Endcap->Fill(disk);
        } else {
          hCluster_Unkn_Endcap->Fill(disk);
        }

        if (posClu.z() > 0) {
          hCluster_Endcap_Fw_XY->Fill(posClu.x(), posClu.y());
          hCluster_Endcap_Fw_RZ_Zoom->Fill(posClu.z(), posClu.perp());
        } else {
          hCluster_Endcap_Bw_XY->Fill(posClu.x(), posClu.y());
          hCluster_Endcap_Bw_RZ_Zoom->Fill(posClu.z(), posClu.perp());
        }
      }

      /// Another way of looking at MC truth
      if (genuineClu) {
        hCluster_Gen_Eta->Fill(fabs(posClu.eta()));
      } else if (combinClu) {
        hCluster_Comb_Eta->Fill(fabs(posClu.eta()));
      } else {
        hCluster_Unkn_Eta->Fill(fabs(posClu.eta()));
      }

      hCluster_PID->Fill(partClu, memberClu);
      hCluster_WidthPhi->Fill(widClu, memberClu);
      hCluster_WidthRZ->Fill(widCluCols, memberClu);

      /// Store Track information in maps, skip if the Cluster is not good
      if (!genuineClu && !combinClu)
        continue;

      std::vector<edm::Ptr<TrackingParticle>> theseTPs = ttClusterAssocHandle->findTrackingParticlePtrs(tempCluRef);

      for (unsigned int i = 0; i < theseTPs.size(); i++) {
        const edm::Ptr<TrackingParticle>& tpPtr = theseTPs.at(i);

        if (tpPtr.isNull())
          continue;

        /// Get the corresponding vertex and reject the track
        /// if its vertex is outside the beampipe
        if (tpPtr->vertex().rho() >= 2.0)
          continue;

        if (barrel) {
          if (tpPerLayer.find(layer) == tpPerLayer.end()) {
            std::vector<edm::Ptr<TrackingParticle>> tempVec;
            tpPerLayer.insert(make_pair(layer, tempVec));
          }
          tpPerLayer[layer].push_back(tpPtr);
        } else {
          if (tpPerDisk.find(disk) == tpPerDisk.end()) {
            std::vector<edm::Ptr<TrackingParticle>> tempVec;
            tpPerDisk.insert(make_pair(disk, tempVec));
          }
          tpPerDisk[disk].push_back(tpPtr);
        }
      }
    }
  }  /// End of Loop over TTClusters

  /// Clean the maps for TrackingParticles and fill histograms
  std::map<unsigned int, std::vector<edm::Ptr<TrackingParticle>>>::iterator iterTPPerLayer;
  std::map<unsigned int, std::vector<edm::Ptr<TrackingParticle>>>::iterator iterTPPerDisk;

  for (iterTPPerLayer = tpPerLayer.begin(); iterTPPerLayer != tpPerLayer.end(); ++iterTPPerLayer) {
    /// Remove duplicates, if any
    std::vector<edm::Ptr<TrackingParticle>> tempVec = iterTPPerLayer->second;
    std::sort(tempVec.begin(), tempVec.end());
    tempVec.erase(std::unique(tempVec.begin(), tempVec.end()), tempVec.end());

    /// Loop over the TrackingParticles in this piece of the map
    for (unsigned int i = 0; i < tempVec.size(); i++) {
      if (tempVec.at(i).isNull())
        continue;
      TrackingParticle thisTP = *(tempVec.at(i));
      mapCluLayer_hTPart_Pt[iterTPPerLayer->first]->Fill(thisTP.p4().pt());
      if (thisTP.p4().pt() > 10.0) {
        mapCluLayer_hTPart_Eta_Pt10[iterTPPerLayer->first]->Fill(thisTP.momentum().eta());
        mapCluLayer_hTPart_Phi_Pt10[iterTPPerLayer->first]->Fill(
            thisTP.momentum().phi() > M_PI ? thisTP.momentum().phi() - 2 * M_PI : thisTP.momentum().phi());
      }
    }
  }

  for (iterTPPerDisk = tpPerDisk.begin(); iterTPPerDisk != tpPerDisk.end(); ++iterTPPerDisk) {
    /// Remove duplicates, if any
    std::vector<edm::Ptr<TrackingParticle>> tempVec = iterTPPerDisk->second;
    std::sort(tempVec.begin(), tempVec.end());
    tempVec.erase(std::unique(tempVec.begin(), tempVec.end()), tempVec.end());

    /// Loop over the TrackingParticles in this piece of the map
    for (unsigned int i = 0; i < tempVec.size(); i++) {
      if (tempVec.at(i).isNull())
        continue;
      TrackingParticle thisTP = *(tempVec.at(i));
      mapCluDisk_hTPart_Pt[iterTPPerDisk->first]->Fill(thisTP.p4().pt());
      if (thisTP.p4().pt() > 10.0) {
        mapCluDisk_hTPart_Eta_Pt10[iterTPPerDisk->first]->Fill(thisTP.momentum().eta());
        mapCluDisk_hTPart_Phi_Pt10[iterTPPerDisk->first]->Fill(reco::reducePhiRange(thisTP.momentum().phi()));
      }
    }
  }

  ////////////////////////////////
  /// COLLECT STUB INFORMATION ///
  ////////////////////////////////

  /// Maps to store TrackingParticle information
  std::map<unsigned int, std::vector<edm::Ptr<TrackingParticle>>> tpPerStubLayer;
  std::map<unsigned int, std::vector<edm::Ptr<TrackingParticle>>> tpPerStubDisk;

  /// Loop over the input Stubs
  typename edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator otherInputIter;
  typename edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator otherContentIter;
  for (otherInputIter = ttStubHandle->begin(); otherInputIter != ttStubHandle->end(); ++otherInputIter) {
    for (otherContentIter = otherInputIter->begin(); otherContentIter != otherInputIter->end(); ++otherContentIter) {
      /// Make the reference to be put in the map
      edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>> tempStubRef =
          edmNew::makeRefTo(ttStubHandle, otherContentIter);

      DetId detIdStub(tempStubRef->getDetId());

      bool genuineStub = ttStubAssocHandle->isGenuine(tempStubRef);
      bool combinStub = ttStubAssocHandle->isCombinatoric(tempStubRef);
      //bool unknownStub    = ttStubAssocHandle->isUnknown( tempStubRef );
      int partStub = 999999999;
      if (genuineStub) {
        edm::Ptr<TrackingParticle> thisTP = ttStubAssocHandle->findTrackingParticlePtr(tempStubRef);
        partStub = thisTP->pdgId();
      }
      double displStub = tempStubRef->rawBend();
      double offsetStub = tempStubRef->bendOffset();

      // Get stub position
      const DetId detId = detIdStub + 1;  // seed sensor
      const GeomDetUnit* det = trackerGeom->idToDetUnit(detId);
      const PixelTopology* pixTopo =
          dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(det)->specificTopology()));
      const Plane& plane = dynamic_cast<const PixelGeomDetUnit*>(det)->surface();
      const MeasurementPoint& mp = tempStubRef->clusterRef(0)->findAverageLocalCoordinatesCentered();
      GlobalPoint posStub = plane.toGlobal(pixTopo->localPosition(mp));

      hStub_RZ->Fill(posStub.z(), posStub.perp());

      bool barrel =
          detIdStub.subdetId() == Phase2Tracker::Subdetector::Barrel || detIdStub.subdetId() == StripSubdetector::TIB;
      unsigned int layer = trackerTopo->layer(detIdStub);
      unsigned int disk = trackerTopo->endcapWheelP2(detIdStub);

      if (barrel) {
        hStub_Barrel->Fill(layer);

        if (genuineStub) {
          hStub_Gen_Barrel->Fill(layer);
        } else if (combinStub) {
          hStub_Comb_Barrel->Fill(layer);
        } else {
          hStub_Unkn_Barrel->Fill(layer);
        }

        hStub_Barrel_XY->Fill(posStub.x(), posStub.y());
        hStub_Barrel_XY_Zoom->Fill(posStub.x(), posStub.y());
      } else {
        // endcap
        hStub_Endcap->Fill(disk);

        if (genuineStub) {
          hStub_Gen_Endcap->Fill(disk);
        } else if (combinStub) {
          hStub_Comb_Endcap->Fill(disk);
        } else {
          hStub_Unkn_Endcap->Fill(disk);
        }

        if (posStub.z() > 0) {
          hStub_Endcap_Fw_XY->Fill(posStub.x(), posStub.y());
          hStub_Endcap_Fw_RZ_Zoom->Fill(posStub.z(), posStub.perp());
        } else {
          hStub_Endcap_Bw_XY->Fill(posStub.x(), posStub.y());
          hStub_Endcap_Bw_RZ_Zoom->Fill(posStub.z(), posStub.perp());
        }
      }

      /// Another way of looking at MC truth
      if (genuineStub) {
        hStub_Gen_Eta->Fill(fabs(posStub.eta()));
      } else if (combinStub) {
        hStub_Comb_Eta->Fill(fabs(posStub.eta()));
      } else {
        hStub_Unkn_Eta->Fill(fabs(posStub.eta()));
      }

      hStub_PID->Fill(partStub);

      /// Store Track information in maps, skip if the Cluster is not good
      if (!genuineStub)
        continue;

      edm::Ptr<TrackingParticle> tpPtr = ttStubAssocHandle->findTrackingParticlePtr(tempStubRef);

      /// Get the corresponding vertex and reject the track
      /// if its vertex is outside the beampipe
      if (tpPtr->vertex().rho() >= 2.0)
        continue;

      if (barrel) {
        if (tpPerStubLayer.find(layer) == tpPerStubLayer.end()) {
          std::vector<edm::Ptr<TrackingParticle>> tempVec;
          tpPerStubLayer.insert(make_pair(layer, tempVec));
        }
        tpPerStubLayer[layer].push_back(tpPtr);

        hStub_Barrel_W->Fill(layer, displStub - offsetStub);
        hStub_Barrel_O->Fill(layer, offsetStub);
      } else {
        // endcap
        if (tpPerStubDisk.find(disk) == tpPerStubDisk.end()) {
          std::vector<edm::Ptr<TrackingParticle>> tempVec;
          tpPerStubDisk.insert(make_pair(disk, tempVec));
        }
        tpPerStubDisk[disk].push_back(tpPtr);

        hStub_Endcap_W->Fill(disk, displStub - offsetStub);
        hStub_Endcap_O->Fill(disk, offsetStub);
      }

      /// Compare to TrackingParticle

      if (tpPtr.isNull())
        continue;  /// This prevents to fill the vector if the TrackingParticle is not found
      const TrackingParticle& thisTP = *tpPtr;

      double simPt = thisTP.p4().pt();
      double simEta = thisTP.momentum().eta();
      double simPhi = reco::reducePhiRange(thisTP.momentum().phi());
      // TO FIX -- replace these old calls with new ones
      /*
      double recPt = theStackedGeometry->findRoughPt( magneticFieldStrength_, &(*tempStubRef) );
      double recEta = theStackedGeometry->findGlobalDirection( &(*tempStubRef) ).eta();
      double recPhi = theStackedGeometry->findGlobalDirection( &(*tempStubRef) ).phi();
      */
      double recPt = 0.;
      double recEta = 0.;
      double recPhi = 0.;

      if (barrel) {
        mapStubLayer_hStub_InvPt_TPart_InvPt[layer]->Fill(1. / simPt, 1. / recPt);
        mapStubLayer_hStub_Pt_TPart_Pt[layer]->Fill(simPt, recPt);
        mapStubLayer_hStub_Eta_TPart_Eta[layer]->Fill(simEta, recEta);
        mapStubLayer_hStub_Phi_TPart_Phi[layer]->Fill(simPhi, recPhi);

        mapStubLayer_hStub_InvPtRes_TPart_Eta[layer]->Fill(simEta, 1. / recPt - 1. / simPt);
        mapStubLayer_hStub_PtRes_TPart_Eta[layer]->Fill(simEta, recPt - simPt);
        mapStubLayer_hStub_EtaRes_TPart_Eta[layer]->Fill(simEta, recEta - simEta);
        mapStubLayer_hStub_PhiRes_TPart_Eta[layer]->Fill(simEta, reco::deltaPhi(recPhi, simPhi));

        mapStubLayer_hStub_W_TPart_Pt[layer]->Fill(simPt, displStub - offsetStub);
        mapStubLayer_hStub_W_TPart_InvPt[layer]->Fill(1. / simPt, displStub - offsetStub);
      } else {
        // endcap
        mapStubDisk_hStub_InvPt_TPart_InvPt[disk]->Fill(1. / simPt, 1. / recPt);
        mapStubDisk_hStub_Pt_TPart_Pt[disk]->Fill(simPt, recPt);
        mapStubDisk_hStub_Eta_TPart_Eta[disk]->Fill(simEta, recEta);
        mapStubDisk_hStub_Phi_TPart_Phi[disk]->Fill(simPhi, recPhi);

        mapStubDisk_hStub_InvPtRes_TPart_Eta[disk]->Fill(simEta, 1. / recPt - 1. / simPt);
        mapStubDisk_hStub_PtRes_TPart_Eta[disk]->Fill(simEta, recPt - simPt);
        mapStubDisk_hStub_EtaRes_TPart_Eta[disk]->Fill(simEta, recEta - simEta);
        mapStubDisk_hStub_PhiRes_TPart_Eta[disk]->Fill(simEta, reco::deltaPhi(recPhi, simPhi));

        mapStubDisk_hStub_W_TPart_Pt[disk]->Fill(simPt, displStub - offsetStub);
        mapStubDisk_hStub_W_TPart_InvPt[disk]->Fill(1. / simPt, displStub - offsetStub);
      }
    }
  }  /// End of loop over TTStubs

  /// Clean the maps for TrackingParticles and fill histograms
  std::map<unsigned int, std::vector<edm::Ptr<TrackingParticle>>>::iterator iterTPPerStubLayer;
  std::map<unsigned int, std::vector<edm::Ptr<TrackingParticle>>>::iterator iterTPPerStubDisk;

  for (iterTPPerStubLayer = tpPerStubLayer.begin(); iterTPPerStubLayer != tpPerStubLayer.end(); ++iterTPPerStubLayer) {
    /// Remove duplicates, if any
    std::vector<edm::Ptr<TrackingParticle>> tempVec = iterTPPerStubLayer->second;
    std::sort(tempVec.begin(), tempVec.end());
    tempVec.erase(std::unique(tempVec.begin(), tempVec.end()), tempVec.end());

    /// Loop over the TrackingParticles in this piece of the map
    for (unsigned int i = 0; i < tempVec.size(); i++) {
      if (tempVec.at(i).isNull())
        continue;
      TrackingParticle thisTP = *(tempVec.at(i));
      mapStubLayer_hTPart_Pt[iterTPPerStubLayer->first]->Fill(thisTP.p4().pt());
      if (thisTP.p4().pt() > 10.0) {
        mapStubLayer_hTPart_Eta_Pt10[iterTPPerStubLayer->first]->Fill(thisTP.momentum().eta());
        mapStubLayer_hTPart_Phi_Pt10[iterTPPerStubLayer->first]->Fill(reco::reducePhiRange(thisTP.momentum().phi()));
      }
    }
  }

  for (iterTPPerStubDisk = tpPerStubDisk.begin(); iterTPPerStubDisk != tpPerStubDisk.end(); ++iterTPPerStubDisk) {
    /// Remove duplicates, if any
    std::vector<edm::Ptr<TrackingParticle>> tempVec = iterTPPerStubDisk->second;
    std::sort(tempVec.begin(), tempVec.end());
    tempVec.erase(std::unique(tempVec.begin(), tempVec.end()), tempVec.end());

    /// Loop over the TrackingParticles in this piece of the map
    for (unsigned int i = 0; i < tempVec.size(); i++) {
      if (tempVec.at(i).isNull())
        continue;
      TrackingParticle thisTP = *(tempVec.at(i));
      mapStubDisk_hTPart_Pt[iterTPPerStubDisk->first]->Fill(thisTP.p4().pt());
      if (thisTP.p4().pt() > 10.0) {
        mapStubDisk_hTPart_Eta_Pt10[iterTPPerStubDisk->first]->Fill(thisTP.momentum().eta());
        mapStubDisk_hTPart_Phi_Pt10[iterTPPerStubDisk->first]->Fill(reco::reducePhiRange(thisTP.momentum().phi()));
      }
    }
  }

  /// //////////////////////////
  /// SPECTRUM OF SIM TRACKS ///
  /// WITHIN PRIMARY VERTEX  ///
  /// CONSTRAINTS            ///
  /// //////////////////////////

  /// Go on only if there are TrackingParticles
  if (!trackingParticleHandle->empty()) {
    /// Loop over TrackingParticles
    std::vector<TrackingParticle>::const_iterator iterTrackingParticles;
    for (iterTrackingParticles = trackingParticleHandle->begin();
         iterTrackingParticles != trackingParticleHandle->end();
         ++iterTrackingParticles) {
      /// Get the corresponding vertex
      /// Assume perfectly round beamspot
      /// Correct and get the correct TrackingParticle Vertex position wrt beam center
      constexpr float d0Cut = 2.;
      if (iterTrackingParticles->vertex().rho() >= d0Cut)
        continue;

      /// First of all, check beamspot and correction
      hSimVtx_XY->Fill(iterTrackingParticles->vertex().x(), iterTrackingParticles->vertex().y());
      hSimVtx_Z->Fill(iterTrackingParticles->vertex().z());

      /// Here we have only tracks form primary vertices
      /// Check Pt spectrum and pseudorapidity for over-threshold tracks
      hTPart_Pt->Fill(iterTrackingParticles->p4().pt());
      if (iterTrackingParticles->p4().pt() > 10.0) {
        hTPart_Eta_Pt10->Fill(iterTrackingParticles->momentum().eta());
        hTPart_Phi_Pt10->Fill(reco::reducePhiRange(iterTrackingParticles->momentum().phi()));
      }
    }  /// End of Loop over TrackingParticles
  }  /// End of if ( TrackingParticleHandle->size() != 0 )

}  /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(AnalyzerClusterStub);
