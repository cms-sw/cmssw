// user includes
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DQMOffline/RecoB/plugins/PrimaryVertexMonitor.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/isFinite.h"

// ROOT includes
#include "TMath.h"

// system includes
#include <fmt/format.h>

using namespace reco;
using namespace edm;

PrimaryVertexMonitor::PrimaryVertexMonitor(const edm::ParameterSet& pSet)
    : vertexInputTag_(pSet.getParameter<InputTag>("vertexLabel")),
      beamSpotInputTag_(pSet.getParameter<InputTag>("beamSpotLabel")),
      vertexToken_(consumes<reco::VertexCollection>(vertexInputTag_)),
      scoreToken_(consumes<VertexScore>(vertexInputTag_)),
      beamspotToken_(consumes<reco::BeamSpot>(beamSpotInputTag_)),
      conf_(pSet),
      TopFolderName_(pSet.getParameter<std::string>("TopFolderName")),
      AlignmentLabel_(pSet.getParameter<std::string>("AlignmentLabel")),
      ndof_(pSet.getParameter<int>("ndof")),
      useHPfoAlignmentPlots_(pSet.getParameter<bool>("useHPforAlignmentPlots")),
      errorPrinted_(false),
      nbvtx(nullptr),
      bsX(nullptr),
      bsY(nullptr),
      bsZ(nullptr),
      bsSigmaZ(nullptr),
      bsDxdz(nullptr),
      bsDydz(nullptr),
      bsBeamWidthX(nullptr),
      bsBeamWidthY(nullptr),
      bsType(nullptr),
      sumpt(nullptr),
      ntracks(nullptr),
      weight(nullptr),
      chi2ndf(nullptr),
      chi2prob(nullptr),
      trackpt(nullptr),
      phi_pt1(nullptr),
      eta_pt1(nullptr),
      phi_pt10(nullptr),
      eta_pt10(nullptr),
      dxy2(nullptr) {}

// -- BeginRun
//---------------------------------------------------------------------------------//
void PrimaryVertexMonitor::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  std::string dqmLabel = "";

  //
  // Book all histograms.
  //

  //  get the store
  dqmLabel = TopFolderName_ + "/" + vertexInputTag_.label();
  iBooker.setCurrentFolder(dqmLabel);

  //   xPos = iBooker.book1D ("xPos","x Coordinate" ,100, -0.1, 0.1);
  auto maxPU = conf_.getParameter<double>("PUMax");
  nbvtx = iBooker.book1D("vtxNbr", "Reconstructed Vertices in Event", maxPU, -0.5, maxPU - 0.5);
  nbgvtx = iBooker.book1D("goodvtxNbr", "Reconstructed Good Vertices in Event", maxPU, -0.5, maxPU - 0.5);

  // to be configured each year...
  auto vposx = conf_.getParameter<double>("Xpos");
  auto vposy = conf_.getParameter<double>("Ypos");

  nbtksinvtx[0] = iBooker.book1D("otherVtxTrksNbr", "Reconstructed Tracks in Vertex (other Vtx)", 40, -0.5, 99.5);
  ntracksVsZ[0] = iBooker.bookProfile(
      "otherVtxTrksVsZ", "Reconstructed Tracks in Vertex (other Vtx) vs Z", 80, -20., 20., 50, 0, 100, "");
  ntracksVsZ[0]->setAxisTitle("z-bs", 1);
  ntracksVsZ[0]->setAxisTitle("#tracks", 2);

  score[0] = iBooker.book1D("otherVtxScore", "sqrt(score) (other Vtx)", 100, 0., 400.);
  trksWeight[0] = iBooker.book1D("otherVtxTrksWeight", "Total weight of Tracks in Vertex (other Vtx)", 40, 0, 100.);
  vtxchi2[0] = iBooker.book1D("otherVtxChi2", "#chi^{2} (other Vtx)", 100, 0., 200.);
  vtxndf[0] = iBooker.book1D("otherVtxNdf", "ndof (other Vtx)", 100, 0., 200.);
  vtxprob[0] = iBooker.book1D("otherVtxProb", "#chi^{2} probability (other Vtx)", 100, 0., 1.);
  nans[0] = iBooker.book1D("otherVtxNans", "Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (other Vtx)", 9, 0.5, 9.5);

  nbtksinvtx[1] = iBooker.book1D("tagVtxTrksNbr", "Reconstructed Tracks in Vertex (tagged Vtx)", 100, -0.5, 99.5);
  ntracksVsZ[1] = iBooker.bookProfile(
      "tagVtxTrksVsZ", "Reconstructed Tracks in Vertex (tagged Vtx) vs Z", 80, -20., 20., 50, 0, 100, "");
  ntracksVsZ[1]->setAxisTitle("z-bs", 1);
  ntracksVsZ[1]->setAxisTitle("#tracks", 2);

  score[1] = iBooker.book1D("tagVtxScore", "sqrt(score) (tagged Vtx)", 100, 0., 400.);
  trksWeight[1] = iBooker.book1D("tagVtxTrksWeight", "Total weight of Tracks in Vertex (tagged Vtx)", 100, 0, 100.);
  vtxchi2[1] = iBooker.book1D("tagVtxChi2", "#chi^{2} (tagged Vtx)", 100, 0., 200.);
  vtxndf[1] = iBooker.book1D("tagVtxNdf", "ndof (tagged Vtx)", 100, 0., 200.);
  vtxprob[1] = iBooker.book1D("tagVtxProb", "#chi^{2} probability (tagged Vtx)", 100, 0., 1.);
  nans[1] = iBooker.book1D("tagVtxNans", "Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (tagged Vtx)", 9, 0.5, 9.5);

  xrec[0] = iBooker.book1D("otherPosX", "Position x Coordinate (other Vtx)", 100, vposx - 0.1, vposx + 0.1);
  yrec[0] = iBooker.book1D("otherPosY", "Position y Coordinate (other Vtx)", 100, vposy - 0.1, vposy + 0.1);
  zrec[0] = iBooker.book1D("otherPosZ", "Position z Coordinate (other Vtx)", 100, -20., 20.);
  xDiff[0] = iBooker.book1D("otherDiffX", "X distance from BeamSpot (other Vtx)", 100, -500, 500);
  yDiff[0] = iBooker.book1D("otherDiffY", "Y distance from BeamSpot (other Vtx)", 100, -500, 500);
  xerr[0] = iBooker.book1D("otherErrX", "Uncertainty x Coordinate (other Vtx)", 100, 0., 100);
  yerr[0] = iBooker.book1D("otherErrY", "Uncertainty y Coordinate (other Vtx)", 100, 0., 100);
  zerr[0] = iBooker.book1D("otherErrZ", "Uncertainty z Coordinate (other Vtx)", 100, 0., 100);
  xerrVsTrks[0] = iBooker.book2D(
      "otherErrVsWeightX", "Uncertainty x Coordinate vs. track weight (other Vtx)", 100, 0, 100., 100, 0., 100);
  yerrVsTrks[0] = iBooker.book2D(
      "otherErrVsWeightY", "Uncertainty y Coordinate vs. track weight (other Vtx)", 100, 0, 100., 100, 0., 100);
  zerrVsTrks[0] = iBooker.book2D(
      "otherErrVsWeightZ", "Uncertainty z Coordinate vs. track weight (other Vtx)", 100, 0, 100., 100, 0., 100);

  xrec[1] = iBooker.book1D("tagPosX", "Position x Coordinate (tagged Vtx)", 100, vposx - 0.1, vposx + 0.1);
  yrec[1] = iBooker.book1D("tagPosY", "Position y Coordinate (tagged Vtx)", 100, vposy - 0.1, vposy + 0.1);
  zrec[1] = iBooker.book1D("tagPosZ", "Position z Coordinate (tagged Vtx)", 100, -20., 20.);
  xDiff[1] = iBooker.book1D("tagDiffX", "X distance from BeamSpot (tagged Vtx)", 100, -500, 500);
  yDiff[1] = iBooker.book1D("tagDiffY", "Y distance from BeamSpot (tagged Vtx)", 100, -500, 500);
  xerr[1] = iBooker.book1D("tagErrX", "Uncertainty x Coordinate (tagged Vtx)", 100, 0., 100);
  yerr[1] = iBooker.book1D("tagErrY", "Uncertainty y Coordinate (tagged Vtx)", 100, 0., 100);
  zerr[1] = iBooker.book1D("tagErrZ", "Uncertainty z Coordinate (tagged Vtx)", 100, 0., 100);
  xerrVsTrks[1] = iBooker.book2D(
      "tagErrVsWeightX", "Uncertainty x Coordinate vs. track weight (tagged Vtx)", 100, 0, 100., 100, 0., 100);
  yerrVsTrks[1] = iBooker.book2D(
      "tagErrVsWeightY", "Uncertainty y Coordinate vs. track weight (tagged Vtx)", 100, 0, 100., 100, 0., 100);
  zerrVsTrks[1] = iBooker.book2D(
      "tagErrVsWeightZ", "Uncertainty z Coordinate vs. track weight (tagged Vtx)", 100, 0, 100., 100, 0., 100);

  type[0] = iBooker.book1D("otherType", "Vertex type (other Vtx)", 3, -0.5, 2.5);
  type[1] = iBooker.book1D("tagType", "Vertex type (tagged Vtx)", 3, -0.5, 2.5);
  for (int i = 0; i < 2; ++i) {
    type[i]->setBinLabel(1, "Valid, real");
    type[i]->setBinLabel(2, "Valid, fake");
    type[i]->setBinLabel(3, "Invalid");
  }

  //  get the store
  dqmLabel = TopFolderName_ + "/" + beamSpotInputTag_.label();
  iBooker.setCurrentFolder(dqmLabel);

  bsX = iBooker.book1D("bsX", "BeamSpot x0", 100, vposx - 0.1, vposx + 0.1);
  bsY = iBooker.book1D("bsY", "BeamSpot y0", 100, vposy - 0.1, vposy + 0.1);
  bsZ = iBooker.book1D("bsZ", "BeamSpot z0", 100, -2., 2.);
  bsSigmaZ = iBooker.book1D("bsSigmaZ", "BeamSpot sigmaZ", 100, 0., 10.);
  bsDxdz = iBooker.book1D("bsDxdz", "BeamSpot dxdz", 100, -0.0003, 0.0003);
  bsDydz = iBooker.book1D("bsDydz", "BeamSpot dydz", 100, -0.0003, 0.0003);
  bsBeamWidthX = iBooker.book1D("bsBeamWidthX", "BeamSpot BeamWidthX", 500, 0., 15.);
  bsBeamWidthY = iBooker.book1D("bsBeamWidthY", "BeamSpot BeamWidthY", 500, 0., 15.);
  bsType = iBooker.book1D("bsType", "BeamSpot type", 4, -1.5, 2.5);
  bsType->setBinLabel(1, "Unknown");
  bsType->setBinLabel(2, "Fake");
  bsType->setBinLabel(3, "LHC");
  bsType->setBinLabel(4, "Tracker");

  //  get the store
  dqmLabel = TopFolderName_ + "/" + AlignmentLabel_;
  iBooker.setCurrentFolder(dqmLabel);

  int TKNoBin = conf_.getParameter<int>("TkSizeBin");
  double TKNoMin = conf_.getParameter<double>("TkSizeMin");
  double TKNoMax = conf_.getParameter<double>("TkSizeMax");

  int DxyBin = conf_.getParameter<int>("DxyBin");
  double DxyMin = conf_.getParameter<double>("DxyMin");
  double DxyMax = conf_.getParameter<double>("DxyMax");

  int PhiBin = conf_.getParameter<int>("PhiBin");
  double PhiMin = conf_.getParameter<double>("PhiMin");
  double PhiMax = conf_.getParameter<double>("PhiMax");

  int EtaBin = conf_.getParameter<int>("EtaBin");
  double EtaMin = conf_.getParameter<double>("EtaMin");
  double EtaMax = conf_.getParameter<double>("EtaMax");

  int PtBin = conf_.getParameter<int>("PtBin");
  double PtMin = conf_.getParameter<double>("PtMin");
  double PtMax = conf_.getParameter<double>("PtMax");

  ntracks = iBooker.book1D("ntracks", "number of PV tracks (p_{T} > 1 GeV)", TKNoBin, TKNoMin, TKNoMax);
  ntracks->setAxisTitle("Number of PV Tracks (p_{T} > 1 GeV) per Event", 1);
  ntracks->setAxisTitle("Number of Event", 2);

  weight = iBooker.book1D("weight", "weight of PV tracks (p_{T} > 1 GeV)", 100, 0., 1.);
  weight->setAxisTitle("weight of PV Tracks (p_{T} > 1 GeV) per Event", 1);
  weight->setAxisTitle("Number of Event", 2);

  sumpt = iBooker.book1D("sumpt", "#Sum p_{T} of PV tracks (p_{T} > 1 GeV)", 100, -0.5, 249.5);
  chi2ndf = iBooker.book1D("chi2ndf", "PV tracks (p_{T} > 1 GeV) #chi^{2}/ndof", 100, 0., 20.);
  chi2prob = iBooker.book1D("chi2prob", "PV tracks (p_{T} > 1 GeV) #chi^{2} probability", 100, 0., 1.);

  dxy2 = iBooker.book1D("dxyzoom", "PV tracks (p_{T} > 1 GeV) d_{xy} (#mum)", DxyBin, DxyMin / 5., DxyMax / 5.);

  trackpt = pvMonitor::makeTH1IfLog(iBooker,
                                    true,
                                    false,
                                    "pt_track",
                                    "PV tracks p_{T};PV tracks p_{T} [GeV];#tracks",
                                    PtBin,
                                    log10(PtMin),
                                    log10(PtMax));

  phi_pt1 = iBooker.book1D("phi_pt1", "PV tracks (p_{T} > 1 GeV) #phi; PV tracks #phi;#tracks", PhiBin, PhiMin, PhiMax);
  eta_pt1 = iBooker.book1D("eta_pt1", "PV tracks (p_{T} > 1 GeV) #eta; PV tracks #eta;#tracks", EtaBin, EtaMin, EtaMax);
  phi_pt10 =
      iBooker.book1D("phi_pt10", "PV tracks (p_{T} > 10 GeV) #phi; PV tracks #phi;#tracks", PhiBin, PhiMin, PhiMax);
  eta_pt10 =
      iBooker.book1D("eta_pt10", "PV tracks (p_{T} > 10 GeV) #phi; PV tracks #eta;#tracks", EtaBin, EtaMin, EtaMax);

  // initialize and book the monitors;
  dxy_pt1.varname_ = "xy";
  dxy_pt1.pTcut_ = 1.f;
  dxy_pt1.bookIPMonitor(iBooker, conf_);

  dxy_pt10.varname_ = "xy";
  dxy_pt10.pTcut_ = 10.f;
  dxy_pt10.bookIPMonitor(iBooker, conf_);

  dz_pt1.varname_ = "z";
  dz_pt1.pTcut_ = 1.f;
  dz_pt1.bookIPMonitor(iBooker, conf_);

  dz_pt10.varname_ = "z";
  dz_pt10.pTcut_ = 10.f;
  dz_pt10.bookIPMonitor(iBooker, conf_);
}

void PrimaryVertexMonitor::IPMonitoring::bookIPMonitor(DQMStore::IBooker& iBooker, const edm::ParameterSet& config) {
  int VarBin = config.getParameter<int>(fmt::format("D{}Bin", varname_));
  double VarMin = config.getParameter<double>(fmt::format("D{}Min", varname_));
  double VarMax = config.getParameter<double>(fmt::format("D{}Max", varname_));

  PhiBin_ = config.getParameter<int>("PhiBin");
  PhiMin_ = config.getParameter<double>("PhiMin");
  PhiMax_ = config.getParameter<double>("PhiMax");
  int PhiBin2D = config.getParameter<int>("PhiBin2D");

  EtaBin_ = config.getParameter<int>("EtaBin");
  EtaMin_ = config.getParameter<double>("EtaMin");
  EtaMax_ = config.getParameter<double>("EtaMax");
  int EtaBin2D = config.getParameter<int>("EtaBin2D");

  PtBin_ = config.getParameter<int>("PtBin");
  PtMin_ = config.getParameter<double>("PtMin") * pTcut_;
  PtMax_ = config.getParameter<double>("PtMax") * pTcut_;

  // 1D variables

  IP_ = iBooker.book1D(fmt::format("d{}_pt{}", varname_, pTcut_),
                       fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_),
                       VarBin,
                       VarMin,
                       VarMax);

  IPErr_ = iBooker.book1D(fmt::format("d{}Err_pt{}", varname_, pTcut_),
                          fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_),
                          100,
                          0.,
                          (varname_.find("xy") != std::string::npos) ? 2000. : 10000.);

  IPPull_ = iBooker.book1D(
      fmt::format("d{}Pull_pt{}", varname_, pTcut_),
      fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}}/#sigma_{{d_{{{}}}}}", pTcut_, varname_, varname_),
      100,
      -5.,
      5.);

  // IP profiles

  IPVsPhi_ = iBooker.bookProfile(fmt::format("d{}VsPhi_pt{}", varname_, pTcut_),
                                 fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track #phi", pTcut_, varname_),
                                 PhiBin_,
                                 PhiMin_,
                                 PhiMax_,
                                 VarBin,
                                 VarMin,
                                 VarMax,
                                 "");
  IPVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 1);
  IPVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 2);

  IPVsEta_ = iBooker.bookProfile(fmt::format("d{}VsEta_pt{}", varname_, pTcut_),
                                 fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track #eta", pTcut_, varname_),
                                 EtaBin_,
                                 EtaMin_,
                                 EtaMax_,
                                 VarBin,
                                 VarMin,
                                 VarMax,
                                 "");
  IPVsEta_->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  IPVsEta_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 2);

  IPVsPt_ = pvMonitor::makeProfileIfLog(
      iBooker,
      true,  /* x-axis */
      false, /* y-axis */
      fmt::format("d{}VsPt_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track p_{{T}}", pTcut_, varname_).c_str(),
      PtBin_,
      log10(PtMin_),
      log10(PtMax_),
      VarMin,
      VarMax,
      "");
  IPVsPt_->setAxisTitle("PV track (p_{T} > 1 GeV) p_{T} [GeV]", 1);
  IPVsPt_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 2);

  // IP error profiles

  IPErrVsPhi_ =
      iBooker.bookProfile(fmt::format("d{}ErrVsPhi_pt{}", varname_, pTcut_),
                          fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #phi", pTcut_, varname_),
                          PhiBin_,
                          PhiMin_,
                          PhiMax_,
                          VarBin,
                          0.,
                          (varname_.find("xy") != std::string::npos) ? 100. : 200.,
                          "");
  IPErrVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 1);
  IPErrVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_), 2);

  IPErrVsEta_ =
      iBooker.bookProfile(fmt::format("d{}ErrVsEta_pt{}", varname_, pTcut_),
                          fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #eta", pTcut_, varname_),
                          EtaBin_,
                          EtaMin_,
                          EtaMax_,
                          VarBin,
                          0.,
                          (varname_.find("xy") != std::string::npos) ? 100. : 200.,
                          "");
  IPErrVsEta_->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  IPErrVsEta_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_), 2);

  IPErrVsPt_ = pvMonitor::makeProfileIfLog(
      iBooker,
      true,  /* x-axis */
      false, /* y-axis */
      fmt::format("d{}ErrVsPt_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track p_{{T}}", pTcut_, varname_).c_str(),
      PtBin_,
      log10(PtMin_),
      log10(PtMax_),
      VarMin,
      VarMax,
      "");
  IPErrVsPt_->setAxisTitle("PV track (p_{T} > 1 GeV) p_{T} [GeV]", 1);
  IPErrVsPt_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_), 2);

  // 2D profiles

  IPVsEtaVsPhi_ = iBooker.bookProfile2D(
      fmt::format("d{}VsEtaVsPhi_pt{}", varname_, pTcut_),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track #eta VS track #phi", pTcut_, varname_),
      EtaBin2D,
      EtaMin_,
      EtaMax_,
      PhiBin2D,
      PhiMin_,
      PhiMax_,
      VarBin,
      VarMin,
      VarMax,
      "");
  IPVsEtaVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  IPVsEtaVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 2);
  IPVsEtaVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 3);

  IPErrVsEtaVsPhi_ = iBooker.bookProfile2D(
      fmt::format("d{}ErrVsEtaVsPhi_pt{}", varname_, pTcut_),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #eta VS track #phi", pTcut_, varname_),
      EtaBin2D,
      EtaMin_,
      EtaMax_,
      PhiBin2D,
      PhiMin_,
      PhiMax_,
      VarBin,
      0.,
      (varname_.find("xy") != std::string::npos) ? 100. : 200.,
      "");
  IPErrVsEtaVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  IPErrVsEtaVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 2);
  IPErrVsEtaVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_),
                                 3);
}

void PrimaryVertexMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(vertexToken_, recVtxs);

  Handle<VertexScore> scores;
  iEvent.getByToken(scoreToken_, scores);

  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken(beamspotToken_, beamSpotHandle);

  //
  // check for absent products and simply "return" in that case
  //
  if (recVtxs.isValid() == false || beamSpotHandle.isValid() == false) {
    edm::LogWarning("PrimaryVertexMonitor")
        << " Some products not available in the event: VertexCollection " << vertexInputTag_ << " " << recVtxs.isValid()
        << " BeamSpot " << beamSpotInputTag_ << " " << beamSpotHandle.isValid() << ". Skipping plots for this event";
    return;
  }

  // check upfront that refs to track are (likely) to be valid
  {
    bool ok = true;
    for (const auto& v : *recVtxs) {
      if (v.tracksSize() > 0) {
        const auto& ref = v.trackRefAt(0);
        if (ref.isNull() || !ref.isAvailable()) {
          if (!errorPrinted_)
            edm::LogWarning("PrimaryVertexMonitor")
                << "Skipping vertex collection: " << vertexInputTag_
                << " since likely the track collection the vertex has refs pointing to is missing (at least the first "
                   "TrackBaseRef is null or not available)";
          else
            errorPrinted_ = true;
          ok = false;
        }
      }
    }
    if (!ok)
      return;
  }

  BeamSpot beamSpot = *beamSpotHandle;

  nbvtx->Fill(recVtxs->size() * 1.);
  int ng = 0;
  for (auto const& vx : (*recVtxs))
    if (vx.isValid() && !vx.isFake() && vx.ndof() >= ndof_)
      ++ng;
  nbgvtx->Fill(ng * 1.);

  if (scores.isValid() && !(*scores).empty()) {
    auto pvScore = (*scores).get(0);
    score[1]->Fill(std::sqrt(pvScore));
    for (unsigned int i = 1; i < (*scores).size(); ++i)
      score[0]->Fill(std::sqrt((*scores).get(i)));
  }

  // fill PV tracks MEs (as now, for alignment)
  if (!recVtxs->empty()) {
    vertexPlots(recVtxs->front(), beamSpot, 1);
    pvTracksPlots(recVtxs->front());

    for (reco::VertexCollection::const_iterator v = recVtxs->begin() + 1; v != recVtxs->end(); ++v)
      vertexPlots(*v, beamSpot, 0);
  }

  // Beamline plots:
  bsX->Fill(beamSpot.x0());
  bsY->Fill(beamSpot.y0());
  bsZ->Fill(beamSpot.z0());
  bsSigmaZ->Fill(beamSpot.sigmaZ());
  bsDxdz->Fill(beamSpot.dxdz());
  bsDydz->Fill(beamSpot.dydz());
  bsBeamWidthX->Fill(beamSpot.BeamWidthX() * cmToUm);
  bsBeamWidthY->Fill(beamSpot.BeamWidthY() * cmToUm);
  bsType->Fill(beamSpot.type());
}

void PrimaryVertexMonitor::pvTracksPlots(const Vertex& v) {
  if (!v.isValid())
    return;
  if (v.isFake())
    return;

  if (v.tracksSize() == 0) {
    ntracks->Fill(0);
    return;
  }

  const math::XYZPoint myVertex(v.position().x(), v.position().y(), v.position().z());

  size_t nTracks = 0;
  float sumPT = 0.;

  for (reco::Vertex::trackRef_iterator t = v.tracks_begin(); t != v.tracks_end(); t++) {
    bool isHighPurity = (**t).quality(reco::TrackBase::highPurity);
    if (!isHighPurity && useHPfoAlignmentPlots_)
      continue;

    float pt = (**t).pt();
    trackpt->Fill(pt);

    if (pt < 1.)
      continue;

    nTracks++;

    float eta = (**t).eta();
    float phi = (**t).phi();

    float w = v.trackWeight(*t);
    float chi2NDF = (**t).normalizedChi2();
    float chi2Prob = TMath::Prob((**t).chi2(), (int)(**t).ndof());
    float Dxy = (**t).dxy(myVertex) * cmToUm;
    float Dz = (**t).dz(myVertex) * cmToUm;
    float DxyErr = (**t).dxyError() * cmToUm;
    float DzErr = (**t).dzError() * cmToUm;

    sumPT += pt * pt;

    // fill MEs
    phi_pt1->Fill(phi);
    eta_pt1->Fill(eta);

    weight->Fill(w);
    chi2ndf->Fill(chi2NDF);
    chi2prob->Fill(chi2Prob);
    dxy2->Fill(Dxy);

    // dxy pT>1

    dxy_pt1.IP_->Fill(Dxy);
    dxy_pt1.IPVsPhi_->Fill(phi, Dxy);
    dxy_pt1.IPVsEta_->Fill(eta, Dxy);
    dxy_pt1.IPVsPt_->Fill(pt, Dxy);
    dxy_pt1.IPVsEtaVsPhi_->Fill(eta, phi, Dxy);

    dxy_pt1.IPErr_->Fill(DxyErr);
    dxy_pt1.IPPull_->Fill(Dxy / DxyErr);
    dxy_pt1.IPErrVsPhi_->Fill(phi, DxyErr);
    dxy_pt1.IPErrVsEta_->Fill(eta, DxyErr);
    dxy_pt1.IPErrVsPt_->Fill(pt, DxyErr);
    dxy_pt1.IPErrVsEtaVsPhi_->Fill(eta, phi, DxyErr);

    // dz pT>1

    dz_pt1.IP_->Fill(Dz);
    dz_pt1.IPVsPhi_->Fill(phi, Dz);
    dz_pt1.IPVsEta_->Fill(eta, Dz);
    dz_pt1.IPVsPt_->Fill(pt, Dz);
    dz_pt1.IPVsEtaVsPhi_->Fill(eta, phi, Dz);

    dz_pt1.IPErr_->Fill(DzErr);
    dz_pt1.IPPull_->Fill(Dz / DzErr);
    dz_pt1.IPErrVsPhi_->Fill(phi, DzErr);
    dz_pt1.IPErrVsEta_->Fill(eta, DzErr);
    dz_pt1.IPErrVsPt_->Fill(pt, DxyErr);
    dz_pt1.IPErrVsEtaVsPhi_->Fill(eta, phi, DzErr);

    if (pt < 10.)
      continue;

    phi_pt10->Fill(phi);
    eta_pt10->Fill(eta);

    // dxy pT>10

    dxy_pt10.IP_->Fill(Dxy);
    dxy_pt10.IPVsPhi_->Fill(phi, Dxy);
    dxy_pt10.IPVsEta_->Fill(eta, Dxy);
    dxy_pt10.IPVsPt_->Fill(pt, Dxy);
    dxy_pt10.IPVsEtaVsPhi_->Fill(eta, phi, Dxy);

    dxy_pt10.IPErr_->Fill(DxyErr);
    dxy_pt10.IPPull_->Fill(Dxy / DxyErr);
    dxy_pt10.IPErrVsPhi_->Fill(phi, DxyErr);
    dxy_pt10.IPErrVsEta_->Fill(eta, DxyErr);
    dxy_pt10.IPErrVsPt_->Fill(pt, DxyErr);
    dxy_pt10.IPErrVsEtaVsPhi_->Fill(eta, phi, DxyErr);

    // dxz pT>10

    dz_pt10.IP_->Fill(Dz);
    dz_pt10.IPVsPhi_->Fill(phi, Dz);
    dz_pt10.IPVsEta_->Fill(eta, Dz);
    dz_pt10.IPVsPt_->Fill(pt, Dz);
    dz_pt10.IPVsEtaVsPhi_->Fill(eta, phi, Dz);

    dz_pt10.IPErr_->Fill(DzErr);
    dz_pt10.IPPull_->Fill(Dz / DzErr);
    dz_pt10.IPErrVsPhi_->Fill(phi, DzErr);
    dz_pt10.IPErrVsEta_->Fill(eta, DzErr);
    dz_pt10.IPErrVsPt_->Fill(pt, DxyErr);
    dz_pt10.IPErrVsEtaVsPhi_->Fill(eta, phi, DzErr);
  }
  ntracks->Fill(float(nTracks));
  sumpt->Fill(sumPT);
}

void PrimaryVertexMonitor::vertexPlots(const Vertex& v, const BeamSpot& beamSpot, int i) {
  if (i < 0 || i > 1)
    return;
  if (!v.isValid())
    type[i]->Fill(2.);
  else if (v.isFake())
    type[i]->Fill(1.);
  else
    type[i]->Fill(0.);

  if (v.isValid() && !v.isFake()) {
    float weight = 0;
    for (reco::Vertex::trackRef_iterator t = v.tracks_begin(); t != v.tracks_end(); t++)
      weight += v.trackWeight(*t);
    trksWeight[i]->Fill(weight);
    nbtksinvtx[i]->Fill(v.tracksSize());
    ntracksVsZ[i]->Fill(v.position().z() - beamSpot.z0(), v.tracksSize());

    vtxchi2[i]->Fill(v.chi2());
    vtxndf[i]->Fill(v.ndof());
    vtxprob[i]->Fill(ChiSquaredProbability(v.chi2(), v.ndof()));

    xrec[i]->Fill(v.position().x());
    yrec[i]->Fill(v.position().y());
    zrec[i]->Fill(v.position().z());

    float xb = beamSpot.x0() + beamSpot.dxdz() * (v.position().z() - beamSpot.z0());
    float yb = beamSpot.y0() + beamSpot.dydz() * (v.position().z() - beamSpot.z0());
    xDiff[i]->Fill((v.position().x() - xb) * cmToUm);
    yDiff[i]->Fill((v.position().y() - yb) * cmToUm);

    xerr[i]->Fill(v.xError() * cmToUm);
    yerr[i]->Fill(v.yError() * cmToUm);
    zerr[i]->Fill(v.zError() * cmToUm);
    xerrVsTrks[i]->Fill(weight, v.xError() * cmToUm);
    yerrVsTrks[i]->Fill(weight, v.yError() * cmToUm);
    zerrVsTrks[i]->Fill(weight, v.zError() * cmToUm);

    nans[i]->Fill(1., edm::isNotFinite(v.position().x()) * 1.);
    nans[i]->Fill(2., edm::isNotFinite(v.position().y()) * 1.);
    nans[i]->Fill(3., edm::isNotFinite(v.position().z()) * 1.);

    int index = 3;
    for (int k = 0; k != 3; k++) {
      for (int j = k; j != 3; j++) {
        index++;
        nans[i]->Fill(index * 1., edm::isNotFinite(v.covariance(k, j)) * 1.);
        // in addition, diagonal element must be positive
        if (j == k && v.covariance(k, j) < 0) {
          nans[i]->Fill(index * 1., 1.);
        }
      }
    }
  }
}

void PrimaryVertexMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("TopFolderName", "OfflinePV");
  desc.add<std::string>("AlignmentLabel", "Alignment");
  desc.add<int>("ndof", 4);
  desc.add<bool>("useHPforAlignmentPlots", true);
  desc.add<InputTag>("vertexLabel", edm::InputTag("offlinePrimaryVertices"));
  desc.add<InputTag>("beamSpotLabel", edm::InputTag("offlineBeamSpot"));
  desc.add<double>("PUMax", 80.0);
  desc.add<double>("Xpos", 0.1);
  desc.add<double>("Ypos", 0.0);
  desc.add<int>("TkSizeBin", 100);
  desc.add<double>("TkSizeMin", -0.5);
  desc.add<double>("TkSizeMax", 499.5);
  desc.add<int>("DxyBin", 100);
  desc.add<double>("DxyMin", -5000.0);
  desc.add<double>("DxyMax", 5000.0);
  desc.add<int>("DzBin", 100);
  desc.add<double>("DzMin", -2000.0);
  desc.add<double>("DzMax", 2000.0);
  desc.add<int>("PhiBin", 32);
  desc.add<double>("PhiMin", -M_PI);
  desc.add<double>("PhiMax", M_PI);
  desc.add<int>("EtaBin", 26);
  desc.add<double>("EtaMin", 2.5);
  desc.add<double>("EtaMax", -2.5);
  desc.add<int>("PtBin", 49);
  desc.add<double>("PtMin", 1.);
  desc.add<double>("PtMax", 50.);
  desc.add<int>("PhiBin2D", 12);
  desc.add<int>("EtaBin2D", 8);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexMonitor);
