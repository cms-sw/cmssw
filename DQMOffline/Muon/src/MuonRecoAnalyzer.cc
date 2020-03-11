#include "DQMOffline/Muon/interface/MuonRecoAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include "TMath.h"
using namespace std;
using namespace edm;

MuonRecoAnalyzer::MuonRecoAnalyzer(const edm::ParameterSet& pSet) {
  parameters = pSet;

  // Input booleans
  IsminiAOD = parameters.getParameter<bool>("IsminiAOD");
  doMVA = parameters.getParameter<bool>("doMVA");
  // the services:
  theMuonCollectionLabel_ = consumes<edm::View<reco::Muon> >(parameters.getParameter<edm::InputTag>("MuonCollection"));
  theVertexLabel_ = consumes<reco::VertexCollection>(pSet.getParameter<InputTag>("inputTagVertex"));
  theBeamSpotLabel_ = consumes<reco::BeamSpot>(pSet.getParameter<InputTag>("inputTagBeamSpot"));
  dcsStatusCollection_ =
      consumes<DcsStatusCollection>(pSet.getUntrackedParameter<std::string>("dcsStatusCollection", "scalersRawToDigi"));

  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");
  pResBin = parameters.getParameter<int>("pResBin");
  pResMin = parameters.getParameter<double>("pResMin");
  pResMax = parameters.getParameter<double>("pResMax");
  rhBin = parameters.getParameter<int>("rhBin");
  rhMin = parameters.getParameter<double>("rhMin");
  rhMax = parameters.getParameter<double>("rhMax");
  pBin = parameters.getParameter<int>("pBin");
  pMin = parameters.getParameter<double>("pMin");
  pMax = parameters.getParameter<double>("pMax");
  chi2Bin = parameters.getParameter<int>("chi2Bin");
  chi2Min = parameters.getParameter<double>("chi2Min");
  chi2Max = parameters.getParameter<double>("chi2Max");
  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");
  tunePBin = parameters.getParameter<int>("tunePBin");
  tunePMax = parameters.getParameter<double>("tunePMax");
  tunePMin = parameters.getParameter<double>("tunePMin");
  thetaBin = parameters.getParameter<int>("thetaBin");
  thetaMin = parameters.getParameter<double>("thetaMin");
  thetaMax = parameters.getParameter<double>("thetaMax");
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");

  theFolder = parameters.getParameter<string>("folder");
}

MuonRecoAnalyzer::~MuonRecoAnalyzer() {}
void MuonRecoAnalyzer::bookHistograms(DQMStore::IBooker& ibooker,
                                      edm::Run const& /*iRun*/,
                                      edm::EventSetup const& /* iSetup */) {
  ibooker.cd();
  ibooker.setCurrentFolder(theFolder);

  muReco = ibooker.book1D("muReco", "muon reconstructed tracks", 6, 1, 7);
  muReco->setBinLabel(1, "glb+tk+sta");
  muReco->setBinLabel(2, "glb+sta");
  muReco->setBinLabel(3, "tk+sta");
  muReco->setBinLabel(4, "tk");
  muReco->setBinLabel(5, "sta");
  muReco->setBinLabel(6, "calo");

  int binFactor = 4;

  /////////////////////////////////////////////////////
  // monitoring of eta parameter
  /////////////////////////////////////////////////////
  std::string histname = "GlbMuon_";
  etaGlbTrack.push_back(ibooker.book1D(histname + "Glb_eta", "#eta_{GLB}", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(ibooker.book1D(histname + "Tk_eta", "#eta_{TKfromGLB}", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(ibooker.book1D(histname + "Sta_eta", "#eta_{STAfromGLB}", etaBin, etaMin, etaMax));
  etaResolution.push_back(ibooker.book1D(
      "Res_TkGlb_eta", "#eta_{TKfromGLB} - #eta_{GLB}", etaBin * binFactor, etaMin / 3000, etaMax / 3000));
  etaResolution.push_back(ibooker.book1D(
      "Res_GlbSta_eta", "#eta_{GLB} - #eta_{STAfromGLB}", etaBin * binFactor, etaMin / 100, etaMax / 100));
  etaResolution.push_back(ibooker.book1D(
      "Res_TkSta_eta", "#eta_{TKfromGLB} - #eta_{STAfromGLB}", etaBin * binFactor, etaMin / 100, etaMax / 100));
  etaResolution.push_back(ibooker.book2D("ResVsEta_TkGlb_eta",
                                         "(#eta_{TKfromGLB} - #eta_{GLB}) vs #eta_{GLB}",
                                         etaBin,
                                         etaMin,
                                         etaMax,
                                         etaBin * binFactor,
                                         etaMin / 3000,
                                         etaMax / 3000));
  etaResolution.push_back(ibooker.book2D("ResVsEta_GlbSta_eta",
                                         "(#eta_{GLB} - #eta_{STAfromGLB}) vs #eta_{GLB}",
                                         etaBin,
                                         etaMin,
                                         etaMax,
                                         etaBin * binFactor,
                                         etaMin / 100,
                                         etaMax / 100));
  etaResolution.push_back(ibooker.book2D("ResVsEta_TkSta_eta",
                                         "(#eta_{TKfromGLB} - #eta_{STAfromGLB}) vs #eta_{TKfromGLB}",
                                         etaBin,
                                         etaMin,
                                         etaMax,
                                         etaBin * binFactor,
                                         etaMin / 100,
                                         etaMax / 100));
  etaPull = ibooker.book1D("Pull_TkSta_eta", "#eta_{TKfromGLB} - #eta_{GLB} / error", 100, -10, 10);
  etaTrack = ibooker.book1D("TkMuon_eta", "#eta_{TK}", etaBin, etaMin, etaMax);
  etaStaTrack = ibooker.book1D("StaMuon_eta", "#eta_{STA}", etaBin, etaMin, etaMax);
  etaEfficiency.push_back(ibooker.book1D("StaEta", "#eta_{STAfromGLB}", etaBin, etaMin, etaMax));
  etaEfficiency.push_back(
      ibooker.book1D("StaEta_ifCombinedAlso", "#eta_{STAfromGLB} if isGlb=true", etaBin, etaMin, etaMax));

  //////////////////////////////////////////////////////
  // monitoring of theta parameter
  /////////////////////////////////////////////////////
  thetaGlbTrack.push_back(ibooker.book1D(histname + "Glb_theta", "#theta_{GLB}", thetaBin, thetaMin, thetaMax));
  thetaGlbTrack[0]->setAxisTitle("rad");
  thetaGlbTrack.push_back(ibooker.book1D(histname + "Tk_theta", "#theta_{TKfromGLB}", thetaBin, thetaMin, thetaMax));
  thetaGlbTrack[1]->setAxisTitle("rad");
  thetaGlbTrack.push_back(ibooker.book1D(histname + "Sta_theta", "#theta_{STAfromGLB}", thetaBin, thetaMin, thetaMax));
  thetaGlbTrack[2]->setAxisTitle("rad");
  thetaResolution.push_back(ibooker.book1D("Res_TkGlb_theta",
                                           "#theta_{TKfromGLB} - #theta_{GLB}",
                                           thetaBin * binFactor,
                                           -(thetaMax / 3000),
                                           thetaMax / 3000));
  thetaResolution[0]->setAxisTitle("rad");
  thetaResolution.push_back(ibooker.book1D("Res_GlbSta_theta",
                                           "#theta_{GLB} - #theta_{STAfromGLB}",
                                           thetaBin * binFactor,
                                           -(thetaMax / 100),
                                           thetaMax / 100));
  thetaResolution[1]->setAxisTitle("rad");
  thetaResolution.push_back(ibooker.book1D("Res_TkSta_theta",
                                           "#theta_{TKfromGLB} - #theta_{STAfromGLB}",
                                           thetaBin * binFactor,
                                           -(thetaMax / 100),
                                           thetaMax / 100));
  thetaResolution[2]->setAxisTitle("rad");
  thetaResolution.push_back(ibooker.book2D("ResVsTheta_TkGlb_theta",
                                           "(#theta_{TKfromGLB} - #theta_{GLB}) vs #theta_{GLB}",
                                           thetaBin,
                                           thetaMin,
                                           thetaMax,
                                           thetaBin * binFactor,
                                           -(thetaMax / 3000),
                                           thetaMax / 3000));
  thetaResolution[3]->setAxisTitle("rad", 1);
  thetaResolution[3]->setAxisTitle("rad", 2);
  thetaResolution.push_back(ibooker.book2D("ResVsTheta_GlbSta_theta",
                                           "(#theta_{GLB} - #theta_{STAfromGLB}) vs #theta_{GLB}",
                                           thetaBin,
                                           thetaMin,
                                           thetaMax,
                                           thetaBin * binFactor,
                                           -(thetaMax / 100),
                                           thetaMax / 100));
  thetaResolution[4]->setAxisTitle("rad", 1);
  thetaResolution[4]->setAxisTitle("rad", 2);
  thetaResolution.push_back(ibooker.book2D("ResVsTheta_TkSta_theta",
                                           "(#theta_{TKfromGLB} - #theta_{STAfromGLB}) vs #theta_{TKfromGLB}",
                                           thetaBin,
                                           thetaMin,
                                           thetaMax,
                                           thetaBin * binFactor,
                                           -(thetaMax / 100),
                                           thetaMax / 100));
  thetaResolution[5]->setAxisTitle("rad", 1);
  thetaResolution[5]->setAxisTitle("rad", 2);
  thetaPull = ibooker.book1D("Pull_TkSta_theta", "#theta_{TKfromGLB} - #theta_{STAfromGLB} / error", 100, -10, 10);
  thetaTrack = ibooker.book1D("TkMuon_theta", "#theta_{TK}", thetaBin, thetaMin, thetaMax);
  thetaTrack->setAxisTitle("rad");
  thetaStaTrack = ibooker.book1D("StaMuon_theta", "#theta_{STA}", thetaBin, thetaMin, thetaMax);
  thetaStaTrack->setAxisTitle("rad");

  // monitoring tunePMuonBestTrack Pt
  tunePResolution = ibooker.book1D(
      "Res_TuneP_pt", "Pt_{MuonBestTrack}-Pt_{tunePMuonBestTrack}/Pt_{MuonBestTrack}", tunePBin, tunePMin, tunePMax);

  // monitoring of phi paramater
  phiGlbTrack.push_back(ibooker.book1D(histname + "Glb_phi", "#phi_{GLB}", phiBin, phiMin, phiMax));
  phiGlbTrack[0]->setAxisTitle("rad");
  phiGlbTrack.push_back(ibooker.book1D(histname + "Tk_phi", "#phi_{TKfromGLB}", phiBin, phiMin, phiMax));
  phiGlbTrack[1]->setAxisTitle("rad");
  phiGlbTrack.push_back(ibooker.book1D(histname + "Sta_phi", "#phi_{STAfromGLB}", phiBin, phiMin, phiMax));
  phiGlbTrack[2]->setAxisTitle("rad");
  phiResolution.push_back(ibooker.book1D(
      "Res_TkGlb_phi", "#phi_{TKfromGLB} - #phi_{GLB}", phiBin * binFactor, phiMin / 3000, phiMax / 3000));
  phiResolution[0]->setAxisTitle("rad");
  phiResolution.push_back(ibooker.book1D(
      "Res_GlbSta_phi", "#phi_{GLB} - #phi_{STAfromGLB}", phiBin * binFactor, phiMin / 100, phiMax / 100));
  phiResolution[1]->setAxisTitle("rad");
  phiResolution.push_back(ibooker.book1D(
      "Res_TkSta_phi", "#phi_{TKfromGLB} - #phi_{STAfromGLB}", phiBin * binFactor, phiMin / 100, phiMax / 100));
  phiResolution[2]->setAxisTitle("rad");
  phiResolution.push_back(ibooker.book2D("ResVsPhi_TkGlb_phi",
                                         "(#phi_{TKfromGLB} - #phi_{GLB}) vs #phi_GLB",
                                         phiBin,
                                         phiMin,
                                         phiMax,
                                         phiBin * binFactor,
                                         phiMin / 3000,
                                         phiMax / 3000));
  phiResolution[3]->setAxisTitle("rad", 1);
  phiResolution[3]->setAxisTitle("rad", 2);
  phiResolution.push_back(ibooker.book2D("ResVsPhi_GlbSta_phi",
                                         "(#phi_{GLB} - #phi_{STAfromGLB}) vs #phi_{GLB}",
                                         phiBin,
                                         phiMin,
                                         phiMax,
                                         phiBin * binFactor,
                                         phiMin / 100,
                                         phiMax / 100));
  phiResolution[4]->setAxisTitle("rad", 1);
  phiResolution[4]->setAxisTitle("rad", 2);
  phiResolution.push_back(ibooker.book2D("ResVsPhi_TkSta_phi",
                                         "(#phi_{TKfromGLB} - #phi_{STAfromGLB}) vs #phi_{TKfromGLB}",
                                         phiBin,
                                         phiMin,
                                         phiMax,
                                         phiBin * binFactor,
                                         phiMin / 100,
                                         phiMax / 100));
  phiResolution[5]->setAxisTitle("rad", 1);
  phiResolution[5]->setAxisTitle("rad", 2);
  phiPull = ibooker.book1D("Pull_TkSta_phi", "#phi_{TKfromGLB} - #phi_{STAfromGLB} / error", 100, -10, 10);
  phiTrack = ibooker.book1D("TkMuon_phi", "#phi_{TK}", phiBin, phiMin, phiMax);
  phiTrack->setAxisTitle("rad");
  phiStaTrack = ibooker.book1D("StaMuon_phi", "#phi_{STA}", phiBin, phiMin, phiMax);
  phiStaTrack->setAxisTitle("rad");
  phiEfficiency.push_back(ibooker.book1D("StaPhi", "#phi_{STAfromGLB}", phiBin, phiMin, phiMax));
  phiEfficiency[0]->setAxisTitle("rad");
  phiEfficiency.push_back(
      ibooker.book1D("StaPhi_ifCombinedAlso", "#phi_{STAfromGLB} if the isGlb=true", phiBin, phiMin, phiMax));
  phiEfficiency[1]->setAxisTitle("rad");

  // monitoring of the chi2 parameter
  chi2OvDFGlbTrack.push_back(
      ibooker.book1D(histname + "Glb_chi2OverDf", "#chi_{2}OverDF_{GLB}", chi2Bin, chi2Min, chi2Max));
  chi2OvDFGlbTrack.push_back(
      ibooker.book1D(histname + "Tk_chi2OverDf", "#chi_{2}OverDF_{TKfromGLB}", phiBin, chi2Min, chi2Max));
  chi2OvDFGlbTrack.push_back(
      ibooker.book1D(histname + "Sta_chi2OverDf", "#chi_{2}OverDF_{STAfromGLB}", chi2Bin, chi2Min, chi2Max));
  chi2OvDFTrack = ibooker.book1D("TkMuon_chi2OverDf", "#chi_{2}OverDF_{TK}", chi2Bin, chi2Min, chi2Max);
  chi2OvDFStaTrack = ibooker.book1D("StaMuon_chi2OverDf", "#chi_{2}OverDF_{STA}", chi2Bin, chi2Min, chi2Max);
  //--------------------------
  probchi2GlbTrack.push_back(ibooker.book1D(histname + "Glb_probchi", "Prob #chi_{GLB}", 120, chi2Min, 1.20));
  probchi2GlbTrack.push_back(ibooker.book1D(histname + "Tk_probchi", "Prob #chi_{TKfromGLB}", 120, chi2Min, 1.20));
  probchi2GlbTrack.push_back(ibooker.book1D(histname + "Sta_probchi", "Prob #chi_{STAfromGLB}", 120, chi2Min, 1.20));
  probchi2Track = ibooker.book1D("TkMuon_probchi", "Prob #chi_{TK}", 120, chi2Min, 1.20);
  probchi2StaTrack = ibooker.book1D("StaMuon_probchi", "Prob #chi_{STA}", 120, chi2Min, 1.20);

  // monitoring of the momentum
  pGlbTrack.push_back(ibooker.book1D(histname + "Glb_p", "p_{GLB}", pBin, pMin, pMax));
  pGlbTrack[0]->setAxisTitle("GeV");
  pGlbTrack.push_back(ibooker.book1D(histname + "Tk_p", "p_{TKfromGLB}", pBin, pMin, pMax));
  pGlbTrack[1]->setAxisTitle("GeV");
  pGlbTrack.push_back(ibooker.book1D(histname + "Sta_p", "p_{STAfromGLB}", pBin, pMin, pMax));
  pGlbTrack[2]->setAxisTitle("GeV");
  pTrack = ibooker.book1D("TkMuon_p", "p_{TK}", pBin, pMin, pMax);
  pTrack->setAxisTitle("GeV");
  pStaTrack = ibooker.book1D("StaMuon_p", "p_{STA}", pBin, pMin, pMax);
  pStaTrack->setAxisTitle("GeV");

  // monitoring of the transverse momentum
  ptGlbTrack.push_back(ibooker.book1D(histname + "Glb_pt", "pt_{GLB}", ptBin, ptMin, ptMax));
  ptGlbTrack[0]->setAxisTitle("GeV");
  ptGlbTrack.push_back(ibooker.book1D(histname + "Tk_pt", "pt_{TKfromGLB}", ptBin, ptMin, ptMax));
  ptGlbTrack[1]->setAxisTitle("GeV");
  ptGlbTrack.push_back(ibooker.book1D(histname + "Sta_pt", "pt_{STAfromGLB}", ptBin, ptMin, ptMax));
  ptGlbTrack[2]->setAxisTitle("GeV");
  ptTrack = ibooker.book1D("TkMuon_pt", "pt_{TK}", ptBin, ptMin, ptMax);
  ptTrack->setAxisTitle("GeV");
  ptStaTrack = ibooker.book1D("StaMuon_pt", "pt_{STA}", ptBin, ptMin, pMax);
  ptStaTrack->setAxisTitle("GeV");

  //monitoring of variables needed by the MVA soft muon

  ptSoftMuonMVA = ibooker.book1D("ptSoftMuonMVA", "pt_{SoftMuon}", 50, 0, 50);
  deltaRSoftMuonMVA = ibooker.book1D("deltaRSoftMuonMVA", "#Delta R", 50, 0, 5);
  gNchi2SoftMuonMVA = ibooker.book1D("gNchi2SoftMuonMVA", "gNchi2", 50, 0, 3);
  vMuHitsSoftMuonMVA = ibooker.book1D("vMuHitsSoftMuonMVA", "vMuHits", 50, 0, 50);
  mNuStationsSoftMuonMVA = ibooker.book1D("mNuStationsSoftMuonMVA", "mNuStations", 6, 0, 6);
  dxyRefSoftMuonMVA = ibooker.book1D("dxyRefSoftMuonMVA", "dxyRef", 50, -0.1, 0.1);
  dzRefSoftMuonMVA = ibooker.book1D("dzRefSoftMuonMVA", "dzRef", 50, -0.1, 0.1);
  LWHSoftMuonMVA = ibooker.book1D("LWHSoftMuonMVA", "LWH", 20, 0, 20);
  valPixHitsSoftMuonMVA = ibooker.book1D("valPixHitsSoftMuonMVA", "valPixHits", 8, 0, 8);
  innerChi2SoftMuonMVA = ibooker.book1D("innerChi2SoftMuonMVA", "innerChi2", 50, 0, 3);
  outerChi2SoftMuonMVA = ibooker.book1D("outerChi2SoftMuonMVA", "outerChi2", 50, 0, 4);
  iValFracSoftMuonMVA = ibooker.book1D("iValFracSoftMuonMVA", "iValFrac", 50, 0.5, 1.0);
  segCompSoftMuonMVA = ibooker.book1D("segCompSoftMuonMVA", "segComp", 50, 0, 1.2);
  chi2LocMomSoftMuonMVA = ibooker.book1D("chi2LocMomSoftMuonMVA", "chi2LocMom", 50, 0, 40);
  chi2LocPosSoftMuonMVA = ibooker.book1D("chi2LocPosSoftMuonMVA", "chi2LocPos", 0, 0, 8);
  glbTrackTailProbSoftMuonMVA = ibooker.book1D("glbTrackTailProbSoftMuonMVA", "glbTrackTailProb", 50, 0, 8);
  NTrkVHitsSoftMuonMVA = ibooker.book1D("NTrkVHitsSoftMuonMVA", "NTrkVHits", 50, 0, 35);
  kinkFinderSoftMuonMVA = ibooker.book1D("kinkFinderSoftMuonMVA", "kinkFinder", 50, 0, 30);
  vRPChitsSoftMuonMVA = ibooker.book1D("vRPChitsSoftMuonMVA", "vRPChits", 50, 0, 50);
  glbKinkFinderSoftMuonMVA = ibooker.book1D("glbKinkFinderSoftMuonMVA", "glbKinkFinder", 50, 0, 50);
  glbKinkFinderLogSoftMuonMVA = ibooker.book1D("glbKinkFinderLogSoftMuonMVA", "glbKinkFinderLog", 50, 0, 50);
  staRelChi2SoftMuonMVA = ibooker.book1D("staRelChi2SoftMuonMVA", "staRelChi2", 50, 0, 2);
  glbDeltaEtaPhiSoftMuonMVA = ibooker.book1D("glbDeltaEtaPhiSoftMuonMVA", "glbDeltaEtaPhi", 50, 0, 0.15);
  trkRelChi2SoftMuonMVA = ibooker.book1D("trkRelChi2SoftMuonMVA", "trkRelChi2", 50, 0, 1.2);
  vDThitsSoftMuonMVA = ibooker.book1D("vDThitsSoftMuonMVA", "vDThits", 50, 0, 50);
  vCSChitsSoftMuonMVA = ibooker.book1D("vCSChitsSoftMuonMVA", "vCSChits", 50, 0, 50);
  timeAtIpInOutSoftMuonMVA = ibooker.book1D("timeAtIpInOutSoftMuonMVA", "timeAtIpInOut", 50, -10.0, 10.0);
  timeAtIpInOutErrSoftMuonMVA = ibooker.book1D("timeAtIpInOutErrSoftMuonMVA", "timeAtIpInOutErr", 50, 0, 3.5);
  getMuonHitsPerStationSoftMuonMVA =
      ibooker.book1D("getMuonHitsPerStationSoftMuonMVA", "getMuonHitsPerStation", 6, 0, 6);
  QprodSoftMuonMVA = ibooker.book1D("QprodSoftMuonMVA", "Qprod", 4, -2, 2);

  // monitoring of the muon charge
  qGlbTrack.push_back(ibooker.book1D(histname + "Glb_q", "q_{GLB}", 5, -2.5, 2.5));
  qGlbTrack.push_back(ibooker.book1D(histname + "Tk_q", "q_{TKfromGLB}", 5, -2.5, 2.5));
  qGlbTrack.push_back(ibooker.book1D(histname + "Sta_q", "q_{STAformGLB}", 5, -2.5, 2.5));
  qGlbTrack.push_back(ibooker.book1D(
      histname + "qComparison", "comparison between q_{GLB} and q_{TKfromGLB}, q_{STAfromGLB}", 8, 0.5, 8.5));
  qGlbTrack[3]->setBinLabel(1, "qGlb=qSta");
  qGlbTrack[3]->setBinLabel(2, "qGlb!=qSta");
  qGlbTrack[3]->setBinLabel(3, "qGlb=qTk");
  qGlbTrack[3]->setBinLabel(4, "qGlb!=qTk");
  qGlbTrack[3]->setBinLabel(5, "qSta=qTk");
  qGlbTrack[3]->setBinLabel(6, "qSta!=qTk");
  qGlbTrack[3]->setBinLabel(7, "qGlb!=qSta,qGlb!=Tk");
  qGlbTrack[3]->setBinLabel(8, "qGlb=qSta,qGlb=Tk");
  qTrack = ibooker.book1D("TkMuon_q", "q_{TK}", 5, -2.5, 2.5);
  qStaTrack = ibooker.book1D("StaMuon_q", "q_{STA}", 5, -2.5, 2.5);

  //////////////////////////////////////////////////////////////
  // monitoring of the momentum resolution
  qOverpResolution.push_back(ibooker.book1D(
      "Res_TkGlb_qOverp", "(q/p)_{TKfromGLB} - (q/p)_{GLB}", pResBin * binFactor * 2, pResMin / 10, pResMax / 10));
  qOverpResolution[0]->setAxisTitle("GeV^{-1}");
  qOverpResolution.push_back(
      ibooker.book1D("Res_GlbSta_qOverp", "(q/p)_{GLB} - (q/p)_{STAfromGLB}", pResBin * binFactor, pResMin, pResMax));
  qOverpResolution[1]->setAxisTitle("GeV^{-1}");
  qOverpResolution.push_back(ibooker.book1D(
      "Res_TkSta_qOverp", "(q/p)_{TKfromGLB} - (q/p)_{STAfromGLB}", pResBin * binFactor, pResMin, pResMax));
  qOverpResolution[2]->setAxisTitle("GeV^{-1}");
  qOverpPull = ibooker.book1D("Pull_TkSta_qOverp", "(q/p)_{TKfromGLB} - (q/p)_{STAfromGLB} / error", 100, -10, 10);

  oneOverpResolution.push_back(ibooker.book1D(
      "Res_TkGlb_oneOverp", "(1/p)_{TKfromGLB} - (1/p)_{GLB}", pResBin * binFactor * 2, pResMin / 10, pResMax / 10));
  oneOverpResolution[0]->setAxisTitle("GeV^{-1}");
  oneOverpResolution.push_back(
      ibooker.book1D("Res_GlbSta_oneOverp", "(1/p)_{GLB} - (1/p)_{STAfromGLB}", pResBin * binFactor, pResMin, pResMax));
  oneOverpResolution[1]->setAxisTitle("GeV^{-1}");
  oneOverpResolution.push_back(ibooker.book1D(
      "Res_TkSta_oneOverp", "(q/p)_{TKfromGLB} - (q/p)_{STAfromGLB}", pResBin * binFactor, pResMin, pResMax));
  oneOverpResolution[2]->setAxisTitle("GeV^{-1}");
  oneOverpPull = ibooker.book1D("Pull_TkSta_oneOverp", "(1/p)_{TKfromGLB} - (1/p)_{STAfromGLB} / error", 100, -10, 10);

  qOverptResolution.push_back(ibooker.book1D("Res_TkGlb_qOverpt",
                                             "(q/p_{t})_{TKfromGLB} - (q/p_{t})_{GLB}",
                                             pResBin * binFactor * 2,
                                             pResMin / 10,
                                             pResMax / 10));
  qOverptResolution[0]->setAxisTitle("GeV^{-1}");
  qOverptResolution.push_back(ibooker.book1D(
      "Res_GlbSta_qOverpt", "(q/p_{t})_{GLB} - (q/p_{t})_{STAfromGLB}", pResBin * binFactor, pResMin, pResMax));
  qOverptResolution[1]->setAxisTitle("GeV^{-1}");
  qOverptResolution.push_back(ibooker.book1D(
      "Res_TkSta_qOverpt", "(q/p_{t})_{TKfromGLB} - (q/p_{t})_{STAfromGLB}", pResBin * binFactor, pResMin, pResMax));
  qOverptResolution[2]->setAxisTitle("GeV^{-1}");
  qOverptPull = ibooker.book1D("Pull_TkSta_qOverpt", "(q/pt)_{TKfromGLB} - (q/pt)_{STAfromGLB} / error", 100, -10, 10);

  oneOverptResolution.push_back(ibooker.book1D("Res_TkGlb_oneOverpt",
                                               "(1/p_{t})_{TKfromGLB} - (1/p_{t})_{GLB}",
                                               pResBin * binFactor * 2,
                                               pResMin / 10,
                                               pResMax / 10));
  oneOverptResolution[0]->setAxisTitle("GeV^{-1}");
  oneOverptResolution.push_back(ibooker.book1D(
      "Res_GlbSta_oneOverpt", "(1/p_{t})_{GLB} - (1/p_{t})_{STAfromGLB}", pResBin * binFactor, pResMin, pResMax));
  oneOverptResolution[1]->setAxisTitle("GeV^{-1}");
  oneOverptResolution.push_back(ibooker.book1D(
      "Res_TkSta_oneOverpt", "(1/p_{t})_{TKfromGLB} - (1/p_{t})_{STAfromGLB}", pResBin * binFactor, pResMin, pResMax));
  oneOverptResolution[2]->setAxisTitle("GeV^{-1}");
  oneOverptResolution.push_back(ibooker.book2D("ResVsEta_TkGlb_oneOverpt",
                                               "(#eta_{TKfromGLB} - #eta_{GLB}) vs (1/p_{t})_{GLB}",
                                               etaBin,
                                               etaMin,
                                               etaMax,
                                               pResBin * binFactor * 2,
                                               pResMin / 10,
                                               pResMax / 10));
  oneOverptResolution[3]->setAxisTitle("GeV^{-1}", 2);
  oneOverptResolution.push_back(ibooker.book2D("ResVsEta_GlbSta_oneOverpt",
                                               "(#eta_{GLB} - #eta_{STAfromGLB} vs (1/p_{t})_{GLB}",
                                               etaBin,
                                               etaMin,
                                               etaMax,
                                               pResBin * binFactor,
                                               pResMin,
                                               pResMax));
  oneOverptResolution[4]->setAxisTitle("GeV^{-1}", 2);
  oneOverptResolution.push_back(ibooker.book2D("ResVsEta_TkSta_oneOverpt",
                                               "(#eta_{TKfromGLB} - #eta_{STAfromGLB}) vs (1/p_{t})_{TKfromGLB}",
                                               etaBin,
                                               etaMin,
                                               etaMax,
                                               pResBin * binFactor,
                                               pResMin,
                                               pResMax));
  oneOverptResolution[5]->setAxisTitle("GeV^{-1}", 2);
  oneOverptResolution.push_back(ibooker.book2D("ResVsPhi_TkGlb_oneOverpt",
                                               "(#phi_{TKfromGLB} - #phi_{GLB}) vs (1/p_{t})_{GLB}",
                                               phiBin,
                                               phiMin,
                                               phiMax,
                                               pResBin * binFactor * 2,
                                               pResMin / 10,
                                               pResMax / 10));
  oneOverptResolution[6]->setAxisTitle("rad", 1);
  oneOverptResolution[6]->setAxisTitle("GeV^{-1}", 2);
  oneOverptResolution.push_back(ibooker.book2D("ResVsPhi_GlbSta_oneOverpt",
                                               "(#phi_{GLB} - #phi_{STAfromGLB} vs (1/p_{t})_{GLB}",
                                               phiBin,
                                               phiMin,
                                               phiMax,
                                               pResBin * binFactor,
                                               pResMin,
                                               pResMax));
  oneOverptResolution[7]->setAxisTitle("rad", 1);
  oneOverptResolution[7]->setAxisTitle("GeV^{-1}", 2);
  oneOverptResolution.push_back(ibooker.book2D("ResVsPhi_TkSta_oneOverpt",
                                               "(#phi_{TKfromGLB} - #phi_{STAfromGLB}) vs (1/p_{t})_{TKfromGLB}",
                                               phiBin,
                                               phiMin,
                                               phiMax,
                                               pResBin * binFactor,
                                               pResMin,
                                               pResMax));
  oneOverptResolution[8]->setAxisTitle("rad", 1);
  oneOverptResolution[8]->setAxisTitle("GeV^{-1}", 2);
  oneOverptResolution.push_back(ibooker.book2D("ResVsPt_TkGlb_oneOverpt",
                                               "((1/p_{t})_{TKfromGLB} - (1/p_{t})_{GLB}) vs (1/p_{t})_{GLB}",
                                               ptBin / 5,
                                               ptMin,
                                               ptMax / 100,
                                               pResBin * binFactor * 2,
                                               pResMin / 10,
                                               pResMax / 10));
  oneOverptResolution[9]->setAxisTitle("GeV^{-1}", 1);
  oneOverptResolution[9]->setAxisTitle("GeV^{-1}", 2);
  oneOverptResolution.push_back(ibooker.book2D("ResVsPt_GlbSta_oneOverpt",
                                               "((1/p_{t})_{GLB} - (1/p_{t})_{STAfromGLB} vs (1/p_{t})_{GLB}",
                                               ptBin / 5,
                                               ptMin,
                                               ptMax / 100,
                                               pResBin * binFactor,
                                               pResMin,
                                               pResMax));
  oneOverptResolution[10]->setAxisTitle("GeV^{-1}", 1);
  oneOverptResolution[10]->setAxisTitle("GeV^{-1}", 2);
  oneOverptResolution.push_back(
      ibooker.book2D("ResVsPt_TkSta_oneOverpt",
                     "((1/p_{t})_{TKfromGLB} - (1/p_{t})_{STAfromGLB}) vs (1/p_{t})_{TKfromGLB}",
                     ptBin / 5,
                     ptMin,
                     ptMax / 100,
                     pResBin * binFactor,
                     pResMin,
                     pResMax));
  oneOverptResolution[11]->setAxisTitle("GeV^{-1}", 1);
  oneOverptResolution[11]->setAxisTitle("GeV^{-1}", 2);
  oneOverptPull =
      ibooker.book1D("Pull_TkSta_oneOverpt", "(1/pt)_{TKfromGLB} - (1/pt)_{STAfromGLB} / error", 100, -10, 10);

  //////////////////////////////////////////////////////////////
  // monitoring of the phi-eta
  phiVsetaGlbTrack.push_back(ibooker.book2D(
      histname + "Glb_phiVSeta", "#phi vs #eta (GLB)", etaBin / 2, etaMin, etaMax, phiBin / 2, phiMin, phiMax));
  phiVsetaGlbTrack.push_back(ibooker.book2D(
      histname + "Tk_phiVSeta", "#phi vs #eta (TKfromGLB)", etaBin / 2, etaMin, etaMax, phiBin / 2, phiMin, phiMax));
  phiVsetaGlbTrack.push_back(ibooker.book2D(
      histname + "Sta_phiVseta", "#phi vs #eta (STAfromGLB)", etaBin / 2, etaMin, etaMax, phiBin / 2, phiMin, phiMax));

  phiVsetaGlbTrack_badlumi.push_back(ibooker.book2D(
      histname + "Glb_phiVSeta_badlumi", "#phi vs #eta (GLB)", etaBin / 2, etaMin, etaMax, phiBin / 2, phiMin, phiMax));
  phiVsetaGlbTrack_badlumi.push_back(ibooker.book2D(histname + "Tk_phiVSeta_badlumi",
                                                    "#phi vs #eta (TKfromGLB)",
                                                    etaBin / 2,
                                                    etaMin,
                                                    etaMax,
                                                    phiBin / 2,
                                                    phiMin,
                                                    phiMax));
  phiVsetaGlbTrack_badlumi.push_back(ibooker.book2D(histname + "Sta_phiVseta_badlumi",
                                                    "#phi vs #eta (STAfromGLB)",
                                                    etaBin / 2,
                                                    etaMin,
                                                    etaMax,
                                                    phiBin / 2,
                                                    phiMin,
                                                    phiMax));

  //////////////////////////////////////////////////////////////
  // monitoring of the recHits provenance
  rhAnalysis.push_back(ibooker.book1D("StaRh_Frac_inGlb", "recHits_{STAinGLB} / recHits_{GLB}", rhBin, rhMin, rhMax));
  rhAnalysis.push_back(ibooker.book1D("TkRh_Frac_inGlb", "recHits_{TKinGLB} / recHits_{GLB}", rhBin, rhMin, rhMax));
  rhAnalysis.push_back(
      ibooker.book1D("StaRh_inGlb_Div_RhAssoSta", "recHits_{STAinGLB} / recHits_{STAfromGLB}", rhBin, rhMin, rhMax));
  rhAnalysis.push_back(
      ibooker.book1D("TkRh_inGlb_Div_RhAssoTk", "recHits_{TKinGLB} / recHits_{TKfromGLB}", rhBin, rhMin, rhMax));
  rhAnalysis.push_back(ibooker.book1D(
      "GlbRh_Div_RhAssoStaTk", "recHits_{GLB} / (recHits_{TKfromGLB}+recHits_{STAfromGLB})", rhBin, rhMin, rhMax));
  rhAnalysis.push_back(ibooker.book1D("invalidRh_Frac_inTk", "Invalid recHits / rechits_{GLB}", rhBin, rhMin, rhMax));

  //////////////////////////////////////////////////////////////
  // monitoring of the muon system rotation w.r.t. tracker
  muVStkSytemRotation.push_back(ibooker.book2D(
      "muVStkSytemRotation_posMu", "pT_{TK} / pT_{GLB} vs pT_{GLB} for #mu^{+}", 50, 0, 200, 100, 0.8, 1.2));
  muVStkSytemRotation.push_back(ibooker.book2D(
      "muVStkSytemRotation_negMu", "pT_{TK} / pT_{GLB} vs pT_{GLB} for #mu^{-}", 50, 0, 200, 100, 0.8, 1.2));
}

void MuonRecoAnalyzer::GetRes(reco::TrackRef t1, reco::TrackRef t2, string par, float& res, float& pull) {
  float p1 = 0, p2 = 0, p1e = 1, p2e = 1;

  if (par == "eta") {
    p1 = t1->eta();
    p1e = t1->etaError();
    p2 = t2->eta();
    p2e = t2->etaError();
  } else if (par == "theta") {
    p1 = t1->theta();
    p1e = t1->thetaError();
    p2 = t2->theta();
    p2e = t2->thetaError();
  } else if (par == "phi") {
    p1 = t1->phi();
    p1e = t1->phiError();
    p2 = t2->phi();
    p2e = t2->phiError();
  } else if (par == "qOverp") {
    p1 = t1->charge() / t1->p();
    p1e = t1->qoverpError();
    p2 = t2->charge() / t2->p();
    p2e = t2->qoverpError();
  } else if (par == "oneOverp") {
    p1 = 1. / t1->p();
    p1e = t1->qoverpError();
    p2 = 1. / t2->p();
    p2e = t2->qoverpError();
  } else if (par == "qOverpt") {
    p1 = t1->charge() / t1->pt();
    p1e = t1->ptError() * p1 * p1;
    p2 = t2->charge() / t2->pt();
    p2e = t2->ptError() * p2 * p2;
  } else if (par == "oneOverpt") {
    p1 = 1. / t1->pt();
    p1e = t1->ptError() * p1 * p1;
    p2 = 1. / t2->pt();
    p2e = t2->ptError() * p2 * p2;
  }

  res = p1 - p2;
  if (p1e != 0 || p2e != 0)
    pull = res / sqrt(p1e * p1e + p2e * p2e);
  else
    pull = -99;
  return;
}

void MuonRecoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogTrace(metname) << "[MuonRecoAnalyzer] Analyze the mu";

  // Take the muon container
  edm::Handle<edm::View<reco::Muon> > muons;
  iEvent.getByToken(theMuonCollectionLabel_, muons);

  Handle<reco::BeamSpot> beamSpot;
  Handle<reco::VertexCollection> vertex;
  if (doMVA) {
    iEvent.getByToken(theBeamSpotLabel_, beamSpot);
    if (!beamSpot.isValid()) {
      edm::LogInfo("MuonRecoAnalyzer") << "Error: Can't get the beamspot" << endl;
      doMVA = false;
    }
    iEvent.getByToken(theVertexLabel_, vertex);
    if (!vertex.isValid()) {
      edm::LogInfo("MuonRecoAnalyzer") << "Error: Can't get the vertex collection" << endl;
      doMVA = false;
    }
  }

  //In this part we determine if we want to fill the plots for events where the DCS flag was set to bad
  edm::Handle<DcsStatusCollection> dcsStatus;
  bool fillBadLumi = false;
  if (iEvent.getByToken(dcsStatusCollection_, dcsStatus) && dcsStatus.isValid()) {
    for (auto const& dcsStatusItr : *dcsStatus) {
      if (!dcsStatusItr.ready(DcsStatus::CSCp))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::CSCm))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::DT0))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::DTp))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::DTm))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::EBp))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::EBm))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::EEp))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::EEm))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::ESp))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::ESm))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::HBHEa))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::HBHEb))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::HBHEc))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::HF))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::HO))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::BPIX))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::FPIX))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::RPC))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::TIBTID))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::TOB))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::TECp))
        fillBadLumi = true;
      if (!dcsStatusItr.ready(DcsStatus::TECm))
        fillBadLumi = true;
      //if (!dcsStatusItr.ready(DcsStatus::CASTOR)) fillBadLumi = true;
    }
  }

  float res = 0, pull = 0;
  if (!muons.isValid())
    return;

  for (edm::View<reco::Muon>::const_iterator muon = muons->begin(); muon != muons->end(); ++muon) {
    //Needed for MVA soft muon

    reco::TrackRef gTrack = muon->globalTrack();
    reco::TrackRef iTrack = muon->innerTrack();
    reco::TrackRef oTrack = muon->outerTrack();
    if (iTrack.isNonnull() && oTrack.isNonnull() && gTrack.isNonnull()) {
      const reco::HitPattern gHits = gTrack->hitPattern();
      const reco::HitPattern iHits = iTrack->hitPattern();
      const reco::MuonQuality muonQuality = muon->combinedQuality();
      int pvIndex = 0;
      math::XYZPoint refPoint;
      if (doMVA) {
        pvIndex = getPv(iTrack.index(), &(*vertex));  //HFDumpUtitilies
        if (pvIndex > -1) {
          refPoint = vertex->at(pvIndex).position();
        } else {
          if (beamSpot.isValid()) {
            refPoint = beamSpot->position();
          } else {
            edm::LogInfo("MuonRecoAnalyzer") << "ERROR: No beam sport found!" << endl;
          }
        }
      }
      ptSoftMuonMVA->Fill(iTrack->eta());
      deltaRSoftMuonMVA->Fill(getDeltaR(*iTrack, *oTrack));
      gNchi2SoftMuonMVA->Fill(gTrack->normalizedChi2());
      vMuHitsSoftMuonMVA->Fill(gHits.numberOfValidMuonHits());
      mNuStationsSoftMuonMVA->Fill(muon->numberOfMatchedStations());
      if (doMVA) {
        dxyRefSoftMuonMVA->Fill(iTrack->dxy(refPoint));
        dzRefSoftMuonMVA->Fill(iTrack->dz(refPoint));
      }
      LWHSoftMuonMVA->Fill(iHits.trackerLayersWithMeasurement());
      valPixHitsSoftMuonMVA->Fill(iHits.numberOfValidPixelHits());
      innerChi2SoftMuonMVA->Fill(iTrack->normalizedChi2());
      outerChi2SoftMuonMVA->Fill(oTrack->normalizedChi2());
      iValFracSoftMuonMVA->Fill(iTrack->validFraction());
      //segCompSoftMuonMVA->Fill(reco::Muon::segmentCompatibility(*muon));
      chi2LocMomSoftMuonMVA->Fill(muonQuality.chi2LocalMomentum);
      chi2LocPosSoftMuonMVA->Fill(muonQuality.chi2LocalPosition);
      glbTrackTailProbSoftMuonMVA->Fill(muonQuality.glbTrackProbability);
      NTrkVHitsSoftMuonMVA->Fill(iHits.numberOfValidTrackerHits());
      kinkFinderSoftMuonMVA->Fill(muonQuality.trkKink);
      vRPChitsSoftMuonMVA->Fill(gHits.numberOfValidMuonRPCHits());
      glbKinkFinderSoftMuonMVA->Fill(muonQuality.glbKink);
      glbKinkFinderLogSoftMuonMVA->Fill(TMath::Log(2 + muonQuality.glbKink));
      staRelChi2SoftMuonMVA->Fill(muonQuality.staRelChi2);
      glbDeltaEtaPhiSoftMuonMVA->Fill(muonQuality.globalDeltaEtaPhi);
      trkRelChi2SoftMuonMVA->Fill(muonQuality.trkRelChi2);
      vDThitsSoftMuonMVA->Fill(gHits.numberOfValidMuonDTHits());
      vCSChitsSoftMuonMVA->Fill(gHits.numberOfValidMuonCSCHits());
      timeAtIpInOutSoftMuonMVA->Fill(muon->time().timeAtIpInOut);
      timeAtIpInOutErrSoftMuonMVA->Fill(muon->time().timeAtIpInOutErr);
      //getMuonHitsPerStationSoftMuonMVA->Fill(gTrack);
      QprodSoftMuonMVA->Fill((iTrack->charge() * oTrack->charge()));
    }

    if (muon->isGlobalMuon()) {
      LogTrace(metname) << "[MuonRecoAnalyzer] The mu is global - filling the histos";
      if (muon->isTrackerMuon() && muon->isStandAloneMuon())
        muReco->Fill(1);
      if (!(muon->isTrackerMuon()) && muon->isStandAloneMuon())
        muReco->Fill(2);
      if (!muon->isStandAloneMuon())
        LogTrace(metname) << "[MuonRecoAnalyzer] ERROR: the mu is global but not standalone!";
      // get the track combinig the information from both the Tracker and the Spectrometer
      reco::TrackRef recoCombinedGlbTrack = muon->combinedMuon();

      // get the track using only the tracker data
      reco::TrackRef recoTkGlbTrack = muon->track();
      // get the track using only the mu spectrometer data
      reco::TrackRef recoStaGlbTrack = muon->standAloneMuon();
      etaGlbTrack[0]->Fill(recoCombinedGlbTrack->eta());
      etaGlbTrack[1]->Fill(recoTkGlbTrack->eta());
      etaGlbTrack[2]->Fill(recoStaGlbTrack->eta());

      phiVsetaGlbTrack[0]->Fill(recoCombinedGlbTrack->eta(), recoCombinedGlbTrack->phi());
      phiVsetaGlbTrack[1]->Fill(recoTkGlbTrack->eta(), recoTkGlbTrack->phi());
      phiVsetaGlbTrack[2]->Fill(recoStaGlbTrack->eta(), recoStaGlbTrack->phi());

      if (fillBadLumi) {
        phiVsetaGlbTrack_badlumi[0]->Fill(recoCombinedGlbTrack->eta(), recoCombinedGlbTrack->phi());
        phiVsetaGlbTrack_badlumi[1]->Fill(recoTkGlbTrack->eta(), recoTkGlbTrack->phi());
        phiVsetaGlbTrack_badlumi[2]->Fill(recoStaGlbTrack->eta(), recoStaGlbTrack->phi());
      }

      GetRes(recoTkGlbTrack, recoCombinedGlbTrack, "eta", res, pull);
      etaResolution[0]->Fill(res);
      GetRes(recoCombinedGlbTrack, recoStaGlbTrack, "eta", res, pull);
      etaResolution[1]->Fill(res);
      GetRes(recoTkGlbTrack, recoStaGlbTrack, "eta", res, pull);
      etaResolution[2]->Fill(res);
      etaPull->Fill(pull);
      etaResolution[3]->Fill(recoCombinedGlbTrack->eta(), recoTkGlbTrack->eta() - recoCombinedGlbTrack->eta());
      etaResolution[4]->Fill(recoCombinedGlbTrack->eta(), -recoStaGlbTrack->eta() + recoCombinedGlbTrack->eta());
      etaResolution[5]->Fill(recoCombinedGlbTrack->eta(), recoTkGlbTrack->eta() - recoStaGlbTrack->eta());

      thetaGlbTrack[0]->Fill(recoCombinedGlbTrack->theta());
      thetaGlbTrack[1]->Fill(recoTkGlbTrack->theta());
      thetaGlbTrack[2]->Fill(recoStaGlbTrack->theta());
      GetRes(recoTkGlbTrack, recoCombinedGlbTrack, "theta", res, pull);
      thetaResolution[0]->Fill(res);
      GetRes(recoCombinedGlbTrack, recoStaGlbTrack, "theta", res, pull);
      thetaResolution[1]->Fill(res);

      GetRes(recoTkGlbTrack, recoStaGlbTrack, "theta", res, pull);
      thetaResolution[2]->Fill(res);
      thetaPull->Fill(pull);
      thetaResolution[3]->Fill(recoCombinedGlbTrack->theta(), recoTkGlbTrack->theta() - recoCombinedGlbTrack->theta());
      thetaResolution[4]->Fill(recoCombinedGlbTrack->theta(),
                               -recoStaGlbTrack->theta() + recoCombinedGlbTrack->theta());
      thetaResolution[5]->Fill(recoCombinedGlbTrack->theta(), recoTkGlbTrack->theta() - recoStaGlbTrack->theta());

      phiGlbTrack[0]->Fill(recoCombinedGlbTrack->phi());
      phiGlbTrack[1]->Fill(recoTkGlbTrack->phi());
      phiGlbTrack[2]->Fill(recoStaGlbTrack->phi());
      GetRes(recoTkGlbTrack, recoCombinedGlbTrack, "phi", res, pull);
      phiResolution[0]->Fill(res);
      GetRes(recoCombinedGlbTrack, recoStaGlbTrack, "phi", res, pull);
      phiResolution[1]->Fill(res);
      GetRes(recoTkGlbTrack, recoStaGlbTrack, "phi", res, pull);
      phiResolution[2]->Fill(res);
      phiPull->Fill(pull);
      phiResolution[3]->Fill(recoCombinedGlbTrack->phi(), recoTkGlbTrack->phi() - recoCombinedGlbTrack->phi());
      phiResolution[4]->Fill(recoCombinedGlbTrack->phi(), -recoStaGlbTrack->phi() + recoCombinedGlbTrack->phi());
      phiResolution[5]->Fill(recoCombinedGlbTrack->phi(), recoTkGlbTrack->phi() - recoStaGlbTrack->phi());

      chi2OvDFGlbTrack[0]->Fill(recoCombinedGlbTrack->normalizedChi2());
      chi2OvDFGlbTrack[1]->Fill(recoTkGlbTrack->normalizedChi2());
      chi2OvDFGlbTrack[2]->Fill(recoStaGlbTrack->normalizedChi2());
      //-------------------------
      //    double probchi = TMath::Prob(recoCombinedGlbTrack->normalizedChi2(),recoCombinedGlbTrack->ndof());
      //    cout << "rellenando histos."<<endl;
      probchi2GlbTrack[0]->Fill(TMath::Prob(recoCombinedGlbTrack->chi2(), recoCombinedGlbTrack->ndof()));
      probchi2GlbTrack[1]->Fill(TMath::Prob(recoTkGlbTrack->chi2(), recoTkGlbTrack->ndof()));
      probchi2GlbTrack[2]->Fill(TMath::Prob(recoStaGlbTrack->chi2(), recoStaGlbTrack->ndof()));
      //    cout << "rellenados histos."<<endl;
      //-------------------------

      pGlbTrack[0]->Fill(recoCombinedGlbTrack->p());
      pGlbTrack[1]->Fill(recoTkGlbTrack->p());
      pGlbTrack[2]->Fill(recoStaGlbTrack->p());

      ptGlbTrack[0]->Fill(recoCombinedGlbTrack->pt());
      ptGlbTrack[1]->Fill(recoTkGlbTrack->pt());
      ptGlbTrack[2]->Fill(recoStaGlbTrack->pt());

      qGlbTrack[0]->Fill(recoCombinedGlbTrack->charge());
      qGlbTrack[1]->Fill(recoTkGlbTrack->charge());
      qGlbTrack[2]->Fill(recoStaGlbTrack->charge());
      if (recoCombinedGlbTrack->charge() == recoStaGlbTrack->charge())
        qGlbTrack[3]->Fill(1);
      else
        qGlbTrack[3]->Fill(2);
      if (recoCombinedGlbTrack->charge() == recoTkGlbTrack->charge())
        qGlbTrack[3]->Fill(3);
      else
        qGlbTrack[3]->Fill(4);
      if (recoStaGlbTrack->charge() == recoTkGlbTrack->charge())
        qGlbTrack[3]->Fill(5);
      else
        qGlbTrack[3]->Fill(6);
      if (recoCombinedGlbTrack->charge() != recoStaGlbTrack->charge() &&
          recoCombinedGlbTrack->charge() != recoTkGlbTrack->charge())
        qGlbTrack[3]->Fill(7);
      if (recoCombinedGlbTrack->charge() == recoStaGlbTrack->charge() &&
          recoCombinedGlbTrack->charge() == recoTkGlbTrack->charge())
        qGlbTrack[3]->Fill(8);

      GetRes(recoTkGlbTrack, recoCombinedGlbTrack, "qOverp", res, pull);
      qOverpResolution[0]->Fill(res);
      GetRes(recoCombinedGlbTrack, recoStaGlbTrack, "qOverp", res, pull);
      qOverpResolution[1]->Fill(res);
      GetRes(recoTkGlbTrack, recoStaGlbTrack, "qOverp", res, pull);
      qOverpResolution[2]->Fill(res);
      qOverpPull->Fill(pull);

      GetRes(recoTkGlbTrack, recoCombinedGlbTrack, "oneOverp", res, pull);
      oneOverpResolution[0]->Fill(res);
      GetRes(recoCombinedGlbTrack, recoStaGlbTrack, "oneOverp", res, pull);
      oneOverpResolution[1]->Fill(res);
      GetRes(recoTkGlbTrack, recoStaGlbTrack, "oneOverp", res, pull);
      oneOverpResolution[2]->Fill(res);
      oneOverpPull->Fill(pull);

      GetRes(recoTkGlbTrack, recoCombinedGlbTrack, "qOverpt", res, pull);
      qOverptResolution[0]->Fill(res);
      GetRes(recoCombinedGlbTrack, recoStaGlbTrack, "qOverpt", res, pull);
      qOverptResolution[1]->Fill(res);
      GetRes(recoTkGlbTrack, recoStaGlbTrack, "qOverpt", res, pull);
      qOverptResolution[2]->Fill(res);
      qOverptPull->Fill(pull);

      GetRes(recoTkGlbTrack, recoCombinedGlbTrack, "oneOverpt", res, pull);
      oneOverptResolution[0]->Fill(res);
      GetRes(recoCombinedGlbTrack, recoStaGlbTrack, "oneOverpt", res, pull);
      oneOverptResolution[1]->Fill(res);
      GetRes(recoTkGlbTrack, recoStaGlbTrack, "oneOverpt", res, pull);
      oneOverptResolution[2]->Fill(res);
      oneOverptPull->Fill(pull);

      // //--- Test new tunePMuonBestTrack() method from Muon.h

      reco::TrackRef recoBestTrack = muon->muonBestTrack();

      reco::TrackRef recoTunePBestTrack = muon->tunePMuonBestTrack();

      double bestTrackPt = recoBestTrack->pt();

      double tunePBestTrackPt = recoTunePBestTrack->pt();

      double tunePBestTrackRes = (bestTrackPt - tunePBestTrackPt) / bestTrackPt;

      tunePResolution->Fill(tunePBestTrackRes);

      oneOverptResolution[3]->Fill(recoCombinedGlbTrack->eta(),
                                   (1 / recoTkGlbTrack->pt()) - (1 / recoCombinedGlbTrack->pt()));
      oneOverptResolution[4]->Fill(recoCombinedGlbTrack->eta(),
                                   -(1 / recoStaGlbTrack->pt()) + (1 / recoCombinedGlbTrack->pt()));
      oneOverptResolution[5]->Fill(recoCombinedGlbTrack->eta(),
                                   (1 / recoTkGlbTrack->pt()) - (1 / recoStaGlbTrack->pt()));
      oneOverptResolution[6]->Fill(recoCombinedGlbTrack->phi(),
                                   (1 / recoTkGlbTrack->pt()) - (1 / recoCombinedGlbTrack->pt()));
      oneOverptResolution[7]->Fill(recoCombinedGlbTrack->phi(),
                                   -(1 / recoStaGlbTrack->pt()) + (1 / recoCombinedGlbTrack->pt()));
      oneOverptResolution[8]->Fill(recoCombinedGlbTrack->phi(),
                                   (1 / recoTkGlbTrack->pt()) - (1 / recoStaGlbTrack->pt()));
      oneOverptResolution[9]->Fill(recoCombinedGlbTrack->pt(),
                                   (1 / recoTkGlbTrack->pt()) - (1 / recoCombinedGlbTrack->pt()));
      oneOverptResolution[10]->Fill(recoCombinedGlbTrack->pt(),
                                    -(1 / recoStaGlbTrack->pt()) + (1 / recoCombinedGlbTrack->pt()));
      oneOverptResolution[11]->Fill(recoCombinedGlbTrack->pt(),
                                    (1 / recoTkGlbTrack->pt()) - (1 / recoStaGlbTrack->pt()));

      if (!IsminiAOD) {
        // valid hits Glb track
        double rhGlb = recoCombinedGlbTrack->found();
        // valid hits Glb track from Tracker
        double rhGlb_StaProvenance = 0;
        // valid hits Glb track from Sta system
        double rhGlb_TkProvenance = 0;

        for (trackingRecHit_iterator recHit = recoCombinedGlbTrack->recHitsBegin();
             recHit != recoCombinedGlbTrack->recHitsEnd();
             ++recHit) {
          if ((*recHit)->isValid()) {
            DetId id = (*recHit)->geographicalId();
            if (id.det() == DetId::Muon)
              rhGlb_StaProvenance++;
            if (id.det() == DetId::Tracker)
              rhGlb_TkProvenance++;
          }
        }
        // valid hits Sta track associated to Glb track
        double rhStaGlb = recoStaGlbTrack->recHitsSize();
        // valid hits Traker track associated to Glb track
        double rhTkGlb = recoTkGlbTrack->found();
        // invalid hits Traker track associated to Glb track
        double rhTkGlb_notValid = recoTkGlbTrack->lost();

        // fill the histos
        rhAnalysis[0]->Fill(rhGlb_StaProvenance / rhGlb);
        rhAnalysis[1]->Fill(rhGlb_TkProvenance / rhGlb);
        rhAnalysis[2]->Fill(rhGlb_StaProvenance / rhStaGlb);
        rhAnalysis[3]->Fill(rhGlb_TkProvenance / rhTkGlb);
        rhAnalysis[4]->Fill(rhGlb / (rhStaGlb + rhTkGlb));
        rhAnalysis[5]->Fill(rhTkGlb_notValid / rhGlb);
      }
      // aligment plots (mu system w.r.t. tracker rotation)
      if (recoCombinedGlbTrack->charge() > 0)
        muVStkSytemRotation[0]->Fill(recoCombinedGlbTrack->pt(), recoTkGlbTrack->pt() / recoCombinedGlbTrack->pt());
      else
        muVStkSytemRotation[1]->Fill(recoCombinedGlbTrack->pt(), recoTkGlbTrack->pt() / recoCombinedGlbTrack->pt());
    }

    if (muon->isTrackerMuon() && !(muon->isGlobalMuon())) {
      LogTrace(metname) << "[MuonRecoAnalyzer] The mu is tracker only - filling the histos";
      if (muon->isStandAloneMuon())
        muReco->Fill(3);
      if (!(muon->isStandAloneMuon()))
        muReco->Fill(4);

      // get the track using only the tracker data
      reco::TrackRef recoTrack = muon->track();

      etaTrack->Fill(recoTrack->eta());
      thetaTrack->Fill(recoTrack->theta());
      phiTrack->Fill(recoTrack->phi());
      chi2OvDFTrack->Fill(recoTrack->normalizedChi2());
      probchi2Track->Fill(TMath::Prob(recoTrack->chi2(), recoTrack->ndof()));
      pTrack->Fill(recoTrack->p());
      ptTrack->Fill(recoTrack->pt());
      qTrack->Fill(recoTrack->charge());
    }

    if (muon->isStandAloneMuon() && !(muon->isGlobalMuon())) {
      LogTrace(metname) << "[MuonRecoAnalyzer] The mu is STA only - filling the histos";
      if (!(muon->isTrackerMuon()))
        muReco->Fill(5);

      // get the track using only the mu spectrometer data
      reco::TrackRef recoStaTrack = muon->standAloneMuon();

      etaStaTrack->Fill(recoStaTrack->eta());
      thetaStaTrack->Fill(recoStaTrack->theta());
      phiStaTrack->Fill(recoStaTrack->phi());
      chi2OvDFStaTrack->Fill(recoStaTrack->normalizedChi2());
      probchi2StaTrack->Fill(TMath::Prob(recoStaTrack->chi2(), recoStaTrack->ndof()));
      pStaTrack->Fill(recoStaTrack->p());
      ptStaTrack->Fill(recoStaTrack->pt());
      qStaTrack->Fill(recoStaTrack->charge());
    }

    if (muon->isCaloMuon() && !(muon->isGlobalMuon()) && !(muon->isTrackerMuon()) && !(muon->isStandAloneMuon()))
      muReco->Fill(6);

    //efficiency plots

    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaGlbTrack = muon->standAloneMuon();

    if (muon->isStandAloneMuon()) {
      etaEfficiency[0]->Fill(recoStaGlbTrack->eta());
      phiEfficiency[0]->Fill(recoStaGlbTrack->phi());
    }
    if (muon->isStandAloneMuon() && muon->isGlobalMuon()) {
      etaEfficiency[1]->Fill(recoStaGlbTrack->eta());
      phiEfficiency[1]->Fill(recoStaGlbTrack->phi());
    }
  }
}

//Needed by MVA Soft Muon
double MuonRecoAnalyzer::getDeltaR(reco::Track track1, reco::Track track2) {
  double dphi = acos(cos(track1.phi() - track2.phi()));
  double deta = track1.eta() - track2.eta();
  return sqrt(dphi * dphi + deta * deta);
}

// ----------------------------------------------------------------------
int MuonRecoAnalyzer::getPv(int tidx, const reco::VertexCollection* vc) {
  if (vc) {
    for (unsigned int i = 0; i < vc->size(); ++i) {
      reco::Vertex::trackRef_iterator v1TrackIter;
      reco::Vertex::trackRef_iterator v1TrackBegin = vc->at(i).tracks_begin();
      reco::Vertex::trackRef_iterator v1TrackEnd = vc->at(i).tracks_end();
      for (v1TrackIter = v1TrackBegin; v1TrackIter != v1TrackEnd; v1TrackIter++) {
        if (static_cast<unsigned int>(tidx) == v1TrackIter->key())
          return i;
      }
    }
  }
  return -1;
}
