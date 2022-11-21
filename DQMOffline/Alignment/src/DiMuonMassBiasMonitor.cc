/*
 *  See header file for a description of this class.
 *
 */

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DQMOffline/Alignment/interface/DiMuonMassBiasMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Histograms/interface/DQMToken.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "TLorentzVector.h"

namespace {
  //constexpr float cmToum = 10e4; /* unused for now */
  constexpr float mumass2 = 0.105658367 * 0.105658367;  //mu mass squared (GeV^2/c^4)
}  // namespace

DiMuonMassBiasMonitor::DiMuonMassBiasMonitor(const edm::ParameterSet& iConfig)
    : ttbESToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("muonTracks"))),
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      MEFolderName_(iConfig.getParameter<std::string>("FolderName")),
      decayMotherName_(iConfig.getParameter<std::string>("decayMotherName")),
      distanceScaleFactor_(iConfig.getParameter<double>("distanceScaleFactor")),
      DiMuMassConfiguration_(iConfig.getParameter<edm::ParameterSet>("DiMuMassConfig")) {}

void DiMuonMassBiasMonitor::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonMassBiasMonitor/MassBias");
  ZMassPlots.bookFromPSet(iBooker, DiMuMassConfiguration_);

  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonMassBiasMonitor");

  // retrieve the mass bins
  const auto& mass_bins = DiMuMassConfiguration_.getParameter<int32_t>("NyBins");
  const auto& mass_min = DiMuMassConfiguration_.getParameter<double>("ymin");
  const auto& mass_max = DiMuMassConfiguration_.getParameter<double>("ymax");

  bookDecayHists(
      iBooker, histosZmm, decayMotherName_, "#mu^{+}#mu^{-}", mass_bins, mass_min, mass_max, distanceScaleFactor_);

  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonMassBiasMonitor/components");
  bookDecayComponentHistograms(iBooker, histosZmm);
}

void DiMuonMassBiasMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::vector<const reco::Track*> myTracks;
  const auto trackHandle = iEvent.getHandle(tracksToken_);
  if (!trackHandle.isValid()) {
    edm::LogError("DiMuonMassBiasMonitor") << "invalid track collection encountered!";
    return;
  }

  for (const auto& muonTrk : *trackHandle) {
    myTracks.emplace_back(&muonTrk);
  }

  if (myTracks.size() != 2) {
    edm::LogWarning("DiMuonMassBiasMonitor") << "There are not enough tracks to monitor!";
    return;
  }

  const auto& t1 = myTracks[1]->momentum();
  const auto& t0 = myTracks[0]->momentum();
  const auto& ditrack = t1 + t0;

  const auto& tplus = myTracks[0]->charge() > 0 ? myTracks[0] : myTracks[1];
  const auto& tminus = myTracks[0]->charge() < 0 ? myTracks[0] : myTracks[1];

  TLorentzVector p4_tplus(tplus->px(), tplus->py(), tplus->pz(), sqrt((tplus->p() * tplus->p()) + mumass2));
  TLorentzVector p4_tminus(tminus->px(), tminus->py(), tminus->pz(), sqrt((tminus->p() * tminus->p()) + mumass2));

  const auto& Zp4 = p4_tplus + p4_tminus;
  float track_invMass = Zp4.M();

  // creat the pair of TLorentVectors used to make the plos
  std::pair<TLorentzVector, TLorentzVector> tktk_p4 = std::make_pair(p4_tplus, p4_tminus);

  // fill the z->mm mass plots
  ZMassPlots.fillPlots(track_invMass, tktk_p4);

  math::XYZPoint ZpT(ditrack.x(), ditrack.y(), 0);
  math::XYZPoint Zp(ditrack.x(), ditrack.y(), ditrack.z());

  // get collection of reconstructed vertices from event
  const auto& vertexHandle = iEvent.getHandle(vertexToken_);
  if (!vertexHandle.isValid()) {
    edm::LogError("DiMuonMassBiasMonitor") << "invalid vertex collection encountered!";
    return;
  }

  // get the vertices from the event
  const auto& vertices = vertexHandle.product();

  // fill the decay vertex plots
  auto decayVertex = fillDecayHistograms(histosZmm, myTracks, vertices, iSetup);

  // get the beamspot from the event
  const reco::BeamSpot* bs;
  const auto& beamSpotHandle = iEvent.getHandle(beamSpotToken_);
  if (!beamSpotHandle.isValid()) {
    bs = nullptr;
  } else {
    bs = beamSpotHandle.product();
  }

  // fill the components plots
  fillComponentHistograms(histosZmm.decayComponents[0], tplus, bs, decayVertex);
  fillComponentHistograms(histosZmm.decayComponents[1], tminus, bs, decayVertex);
}

void DiMuonMassBiasMonitor::bookDecayComponentHistograms(DQMStore::IBooker& ibook, DecayHists& histos) const {
  bookComponentHists(ibook, histos, "mu_plus", 0.1);
  bookComponentHists(ibook, histos, "mu_minus", 0.1);
}

void DiMuonMassBiasMonitor::bookComponentHists(DQMStore::IBooker& ibook,
                                               DecayHists& histos,
                                               TString const& componentName,
                                               float distanceScaleFactor) const {
  ComponentHists comp;

  comp.h_pt = ibook.book1D(componentName + "_pt", "track momentum ;p_{T} [GeV]", 100, 0., 100.);
  comp.h_eta = ibook.book1D(componentName + "_eta", "track rapidity;#eta", 100, -2.5, 2.5);
  comp.h_phi = ibook.book1D(componentName + "_phi", "track azimuth;#phi", 100, -M_PI, M_PI);
  comp.h_dxy = ibook.book1D(componentName + "_dxyBS",
                            "TIP w.r.t BS;d_{xy}(BS) [cm]",
                            100,
                            -0.5 * distanceScaleFactor,
                            0.5 * distanceScaleFactor);
  comp.h_exy =
      ibook.book1D(componentName + "_exy", "Error on TIP ;#sigma(d_{xy}(BS)) [cm]", 100, 0, 0.05 * distanceScaleFactor);
  comp.h_dz = ibook.book1D(componentName + "_dzPV",
                           "LIP w.r.t PV;d_{z}(PV) [cm]",
                           100,
                           -0.5 * distanceScaleFactor,
                           0.5 * distanceScaleFactor);
  comp.h_ez =
      ibook.book1D(componentName + "_ez", "Error on LIP;#sigma(d_{z}(PV)) [cm]", 100, 0, 0.5 * distanceScaleFactor);
  comp.h_chi2 = ibook.book1D(componentName + "_chi2", ";#chi^{2}", 100, 0, 20);

  histos.decayComponents.push_back(comp);
}

void DiMuonMassBiasMonitor::bookDecayHists(DQMStore::IBooker& ibook,
                                           DecayHists& decayHists,
                                           std::string const& name,
                                           std::string const& products,
                                           int nMassBins,
                                           float massMin,
                                           float massMax,
                                           float distanceScaleFactor) const {
  std::string histTitle = name + " #rightarrow " + products + ";";

  decayHists.h_mass = ibook.book1D("h_mass", histTitle + "M(" + products + ") [GeV]", nMassBins, massMin, massMax);
  decayHists.h_pt = ibook.book1D("h_pt", histTitle + "p_{T} [GeV]", 100, 0.00, 200.0);
  decayHists.h_eta = ibook.book1D("h_eta", histTitle + "#eta", 100, -5., 5.);
  decayHists.h_phi = ibook.book1D("h_phi", histTitle + "#varphi [rad]", 100, -M_PI, M_PI);
  decayHists.h_displ2D =
      ibook.book1D("h_displ2D", histTitle + "vertex 2D displacement [cm]", 100, 0.00, 0.1 * distanceScaleFactor);
  decayHists.h_sign2D =
      ibook.book1D("h_sign2D", histTitle + "vertex 2D displ. significance", 100, 0.00, 100.0 * distanceScaleFactor);
  decayHists.h_ct = ibook.book1D("h_ct", histTitle + "c#tau [cm]", 100, 0.00, 0.4 * distanceScaleFactor);
  decayHists.h_pointing = ibook.book1D("h_pointing", histTitle + "cos( 2D pointing angle )", 100, -1, 1);
  decayHists.h_vertNormChi2 = ibook.book1D("h_vertNormChi2", histTitle + "vertex #chi^{2}/ndof", 100, 0.00, 10);
  decayHists.h_vertProb = ibook.book1D("h_vertProb", histTitle + "vertex prob.", 100, 0.00, 1.0);
}

reco::Vertex const* DiMuonMassBiasMonitor::fillDecayHistograms(DecayHists const& histos,
                                                               std::vector<const reco::Track*> const& tracks,
                                                               const reco::VertexCollection* const& pvs,
                                                               const edm::EventSetup& iSetup) const {
  if (tracks.size() != 2) {
    edm::LogWarning("DiMuonVertexMonitor") << "There are not enough tracks to construct a vertex!";
    return nullptr;
  }

  const TransientTrackBuilder* theB = &iSetup.getData(ttbESToken_);
  TransientVertex mumuTransientVtx;
  std::vector<reco::TransientTrack> tks;

  for (const auto& track : tracks) {
    reco::TransientTrack trajectory = theB->build(track);
    tks.push_back(trajectory);
  }

  KalmanVertexFitter kalman(true);
  mumuTransientVtx = kalman.vertex(tks);

  auto svtx = reco::Vertex(mumuTransientVtx);
  if (not svtx.isValid()) {
    return nullptr;
  }

  const auto& mom_t1 = tracks[1]->momentum();
  const auto& mom_t0 = tracks[0]->momentum();
  const auto& momentum = mom_t1 + mom_t0;

  TLorentzVector p4_t0(mom_t0.x(), mom_t0.y(), mom_t0.z(), sqrt(mom_t0.mag2() + mumass2));
  TLorentzVector p4_t1(mom_t1.x(), mom_t1.y(), mom_t1.z(), sqrt(mom_t1.mag2() + mumass2));

  const auto& p4 = p4_t0 + p4_t1;
  float mass = p4.M();

  auto pvtx = std::min_element(pvs->begin(), pvs->end(), [svtx](reco::Vertex const& pv1, reco::Vertex const& pv2) {
    return abs(pv1.z() - svtx.z()) < abs(pv2.z() - svtx.z());
  });

  if (pvtx == pvs->end()) {
    return nullptr;
  }

  VertexDistanceXY vdistXY;
  Measurement1D distXY = vdistXY.distance(svtx, *pvtx);

  auto pvtPos = pvtx->position();
  const auto& svtPos = svtx.position();

  math::XYZVector displVect2D(svtPos.x() - pvtPos.x(), svtPos.y() - pvtPos.y(), 0);
  auto cosAlpha = displVect2D.Dot(momentum) / (displVect2D.Rho() * momentum.rho());

  auto ct = distXY.value() * cosAlpha * mass / momentum.rho();

  histos.h_pointing->Fill(cosAlpha);

  histos.h_mass->Fill(mass);

  histos.h_pt->Fill(momentum.rho());
  histos.h_eta->Fill(momentum.eta());
  histos.h_phi->Fill(momentum.phi());

  histos.h_ct->Fill(ct);

  histos.h_displ2D->Fill(distXY.value());
  histos.h_sign2D->Fill(distXY.significance());

  if (svtx.chi2() >= 0) {
    histos.h_vertNormChi2->Fill(svtx.chi2() / svtx.ndof());
    histos.h_vertProb->Fill(ChiSquaredProbability(svtx.chi2(), svtx.ndof()));
  }

  return &*pvtx;
}

void DiMuonMassBiasMonitor::fillComponentHistograms(ComponentHists const& histos,
                                                    const reco::Track* const& component,
                                                    reco::BeamSpot const* bs,
                                                    reco::Vertex const* pv) const {
  histos.h_pt->Fill(component->pt());
  histos.h_eta->Fill(component->eta());
  histos.h_phi->Fill(component->phi());

  math::XYZPoint zero(0, 0, 0);
  math::Error<3>::type zeroCov;  // needed for dxyError
  if (bs) {
    histos.h_dxy->Fill(component->dxy(*bs));
    histos.h_exy->Fill(component->dxyError(*bs));
  } else {
    histos.h_dxy->Fill(component->dxy(zero));
    histos.h_exy->Fill(component->dxyError(zero, zeroCov));
  }
  if (pv) {
    histos.h_dz->Fill(component->dz(pv->position()));
  } else {
    histos.h_dz->Fill(component->dz(zero));
  }
  histos.h_ez->Fill(component->dzError());

  histos.h_chi2->Fill(component->chi2() / component->ndof());
}

void DiMuonMassBiasMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonTracks", edm::InputTag("ALCARECOTkAlDiMuon"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("FolderName", "DiMuonMassBiasMonitor");
  desc.add<std::string>("decayMotherName", "Z");
  desc.add<double>("distanceScaleFactor", 0.1);

  {
    edm::ParameterSetDescription psDiMuMass;
    psDiMuMass.add<std::string>("name", "DiMuMass");
    psDiMuMass.add<std::string>("title", "M(#mu#mu)");
    psDiMuMass.add<std::string>("yUnits", "[GeV]");
    psDiMuMass.add<int>("NxBins", 24);
    psDiMuMass.add<int>("NyBins", 50);
    psDiMuMass.add<double>("ymin", 65.);
    psDiMuMass.add<double>("ymax", 115.);
    psDiMuMass.add<double>("maxDeltaEta", 3.7);
    desc.add<edm::ParameterSetDescription>("DiMuMassConfig", psDiMuMass);
  }

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(DiMuonMassBiasMonitor);
