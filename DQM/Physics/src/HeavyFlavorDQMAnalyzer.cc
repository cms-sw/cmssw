#include "DQM/Physics/src/HeavyFlavorDQMAnalyzer.h"

void throwMissingCollection(const char* requested, const char* missing) {
  throw cms::Exception("Configuration") << "Requested plots for from the collection " << requested
                                        << " also requires a collection in " << missing << std::endl;
}

float getMass(pat::CompositeCandidate const& cand) {
  float mass = cand.mass();
  if (cand.hasUserFloat("fitMass")) {
    mass = cand.userFloat("fitMass");
  }
  return mass;
}

//
// constructors and destructor
//
HeavyFlavorDQMAnalyzer::HeavyFlavorDQMAnalyzer(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      pvCollectionToken(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("pvCollection"))),
      beamSpotToken(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))) {
  if (iConfig.existsAs<edm::InputTag>("OniaToMuMuCands")) {
    oniaToMuMuCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("OniaToMuMuCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("BuToJPsiKCands")) {
    if (oniaToMuMuCandsToken.isUninitialized())
      throwMissingCollection("BuToJPsiKCands", "OniaToMuMuCands");
    buToJPsiKCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("BuToJPsiKCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("Kx0ToKPiCands")) {
    kx0ToKPiCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("Kx0ToKPiCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("BdToJPsiKx0Cands")) {
    if (oniaToMuMuCandsToken.isUninitialized())
      throwMissingCollection("BdToJPsiKx0Cands", "OniaToMuMuCands");
    if (kx0ToKPiCandsToken.isUninitialized())
      throwMissingCollection("BdToJPsiKx0Cands", "Kx0ToKPiCands");
    bdToJPsiKx0CandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("BdToJPsiKx0Cands"));
  }
  if (iConfig.existsAs<edm::InputTag>("PhiToKKCands")) {
    phiToKKCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("PhiToKKCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("BsToJPsiPhiCands")) {
    if (oniaToMuMuCandsToken.isUninitialized())
      throwMissingCollection("BsToJPsiPhiCands", "OniaToMuMuCands");
    if (phiToKKCandsToken.isUninitialized())
      throwMissingCollection("BsToJPsiPhiCands", "PhiToKKCands");
    bsToJPsiPhiCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("BsToJPsiPhiCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("K0sToPiPiCands")) {
    k0sToPiPiCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("K0sToPiPiCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("BdToJPsiK0sCands")) {
    if (oniaToMuMuCandsToken.isUninitialized())
      throwMissingCollection("BdToJPsiK0sCands", "OniaToMuMuCands");
    if (k0sToPiPiCandsToken.isUninitialized())
      throwMissingCollection("BdToJPsiK0sCands", "K0sToPiPiCands");
    bdToJPsiK0sCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("BdToJPsiK0sCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("Lambda0ToPPiCands")) {
    lambda0ToPPiCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("Lambda0ToPPiCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("LambdaBToJPsiLambda0Cands")) {
    if (oniaToMuMuCandsToken.isUninitialized())
      throwMissingCollection("LambdaBToJPsiLambda0Cands", "OniaToMuMuCands");
    if (lambda0ToPPiCandsToken.isUninitialized())
      throwMissingCollection("LambdaBToJPsiLambda0Cands", "Lambda0ToPPiCands");
    lambdaBToJPsiLambda0CandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("LambdaBToJPsiLambda0Cands"));
  }
  if (iConfig.existsAs<edm::InputTag>("BcToJPsiPiCands")) {
    if (oniaToMuMuCandsToken.isUninitialized())
      throwMissingCollection("BcToJPsiPiCands", "OniaToMuMuCands");
    bcToJPsiPiCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("BcToJPsiPiCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("Psi2SToJPsiPiPiCands")) {
    if (oniaToMuMuCandsToken.isUninitialized())
      throwMissingCollection("Psi2SToJPsiPiPiCands", "OniaToMuMuCands");
    psi2SToJPsiPiPiCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("Psi2SToJPsiPiPiCands"));
  }
  if (iConfig.existsAs<edm::InputTag>("BuToPsi2SKCands")) {
    if (psi2SToJPsiPiPiCandsToken.isUninitialized())
      throwMissingCollection("BuToPsi2SKCands", "Psi2SToJPsiPiPiCands");
    buToPsi2SKCandsToken =
        consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("BuToPsi2SKCands"));
  }
}

HeavyFlavorDQMAnalyzer::~HeavyFlavorDQMAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------

void HeavyFlavorDQMAnalyzer::dqmAnalyze(edm::Event const& iEvent,
                                        edm::EventSetup const& iSetup,
                                        Histograms const& histos) const {
  auto& pvColl = iEvent.get(pvCollectionToken);
  auto bs = iEvent.getHandle(beamSpotToken).product();

  std::vector<bool> displacedJPsiToMuMu;
  pat::CompositeCandidateCollection lOniaToMuMu;
  if (not oniaToMuMuCandsToken.isUninitialized()) {
    lOniaToMuMu = iEvent.get(oniaToMuMuCandsToken);
    displacedJPsiToMuMu.resize(lOniaToMuMu.size(), false);
  }

  if (not buToJPsiKCandsToken.isUninitialized()) {
    auto lBuToJPsiK = iEvent.get(buToJPsiKCandsToken);
    for (auto&& cand : lBuToJPsiK) {
      auto jpsi = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi");
      auto jpsiMass = getMass(*jpsi);
      if (jpsiMass < 2.9 or jpsiMass > 3.3)
        continue;

      auto closestPV = fillDecayHistograms(histos.buToJPsiK, cand, pvColl);
      if (not closestPV)
        continue;
      fillBuToJPsiKComponents(histos.buToJPsiK, cand, bs, closestPV);

      displacedJPsiToMuMu[jpsi.index()] = true;
    }
  }

  std::vector<bool> displacedPsi2SToJPsiPiPi;
  pat::CompositeCandidateCollection lPsi2SToJPsiPiPi;
  if (not psi2SToJPsiPiPiCandsToken.isUninitialized()) {
    lPsi2SToJPsiPiPi = iEvent.get(psi2SToJPsiPiPiCandsToken);
    displacedPsi2SToJPsiPiPi.resize(lPsi2SToJPsiPiPi.size(), false);
  }

  if (not buToPsi2SKCandsToken.isUninitialized()) {
    auto lBuToPsi2SK = iEvent.get(buToPsi2SKCandsToken);
    for (auto&& cand : lBuToPsi2SK) {
      auto psi2S = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToPsi2S");
      auto psi2SMass = getMass(*psi2S);
      if (psi2SMass < 3.65 or psi2SMass > 3.72)
        continue;

      auto jpsi = *psi2S->userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi");
      auto jpsiMass = getMass(*jpsi);
      if (jpsiMass < 2.9 or jpsiMass > 3.3)
        continue;

      auto closestPV = fillDecayHistograms(histos.buToPsi2SK, cand, pvColl);
      if (not closestPV)
        continue;
      fillBuToPsi2SKComponents(histos.buToPsi2SK, cand, bs, closestPV);

      displacedPsi2SToJPsiPiPi[psi2S.index()] = true;
      displacedJPsiToMuMu[jpsi.index()] = true;
    }
  }

  for (size_t i = 0; i < lPsi2SToJPsiPiPi.size(); i++) {
    auto&& cand = lPsi2SToJPsiPiPi[i];

    auto jpsi = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi");
    auto jpsiMass = getMass(*jpsi);
    if (jpsiMass < 2.9 or jpsiMass > 3.3)
      continue;

    auto closestPV = fillDecayHistograms(histos.psi2SToJPsiPiPi, cand, pvColl);
    if (not closestPV)
      continue;
    fillPsi2SToJPsiPiPiComponents(histos.psi2SToJPsiPiPi, cand, bs, closestPV);

    auto decayHistos = &histos.psi2SToJPsiPiPiPrompt;
    if (displacedPsi2SToJPsiPiPi[i]) {
      decayHistos = &histos.psi2SToJPsiPiPiDispl;
    }

    fillDecayHistograms(*decayHistos, cand, pvColl);
    fillPsi2SToJPsiPiPiComponents(*decayHistos, cand, bs, closestPV);
  }
  lPsi2SToJPsiPiPi.clear();
  displacedPsi2SToJPsiPiPi.clear();

  std::vector<bool> displacedKx0ToKPi;
  pat::CompositeCandidateCollection lKx0ToKPi;
  if (not kx0ToKPiCandsToken.isUninitialized()) {
    lKx0ToKPi = iEvent.get(kx0ToKPiCandsToken);
    displacedKx0ToKPi.resize(lKx0ToKPi.size(), false);
  }

  if (not bdToJPsiKx0CandsToken.isUninitialized()) {
    auto lBdToJPsiKx0 = iEvent.get(bdToJPsiKx0CandsToken);
    for (auto&& cand : lBdToJPsiKx0) {
      auto jpsi = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi");
      auto jpsiMass = getMass(*jpsi);
      if (jpsiMass < 2.9 or jpsiMass > 3.3)
        continue;

      auto kx0 = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToKx0");
      auto kx0Mass = getMass(*kx0);
      if (kx0Mass < 0.77 or kx0Mass > 1.02)
        continue;

      auto closestPV = fillDecayHistograms(histos.bdToJPsiKx0, cand, pvColl);
      if (not closestPV)
        continue;
      fillBdToJPsiKx0Components(histos.bdToJPsiKx0, cand, bs, closestPV);

      displacedKx0ToKPi[kx0.index()] = true;
    }
  }

  for (size_t i = 0; i < lKx0ToKPi.size(); i++) {
    auto&& cand = lKx0ToKPi[i];

    auto closestPV = fillDecayHistograms(histos.kx0ToKPi, cand, pvColl);
    if (not closestPV)
      continue;
    fillKx0ToKPiComponents(histos.kx0ToKPi, cand, bs, closestPV);

    auto decayHistos = &histos.kx0ToKPiPrompt;
    if (displacedKx0ToKPi[i]) {
      decayHistos = &histos.kx0ToKPiDispl;
    }

    fillDecayHistograms(*decayHistos, cand, pvColl);
    fillKx0ToKPiComponents(*decayHistos, cand, bs, closestPV);
  }
  lKx0ToKPi.clear();
  displacedKx0ToKPi.clear();

  std::vector<bool> displacedPhiToKK;
  pat::CompositeCandidateCollection lPhiToKK;
  if (not phiToKKCandsToken.isUninitialized()) {
    lPhiToKK = iEvent.get(phiToKKCandsToken);
    displacedPhiToKK.resize(lPhiToKK.size(), false);
  }

  if (not bsToJPsiPhiCandsToken.isUninitialized()) {
    auto lBsToJPsiPhi = iEvent.get(bsToJPsiPhiCandsToken);
    for (auto&& cand : lBsToJPsiPhi) {
      auto jpsi = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi");
      auto jpsiMass = getMass(*jpsi);
      if (jpsiMass < 2.9 or jpsiMass > 3.3)
        continue;

      auto phi = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToPhi");
      auto phiMass = getMass(*phi);
      if (phiMass < 1.005 or phiMass > 1.035)
        continue;

      auto closestPV = fillDecayHistograms(histos.bsToJPsiPhi, cand, pvColl);
      if (not closestPV)
        continue;
      fillBsToJPsiPhiComponents(histos.bsToJPsiPhi, cand, bs, closestPV);

      displacedJPsiToMuMu[jpsi.index()] = true;
      displacedPhiToKK[phi.index()] = true;
    }
  }

  for (size_t i = 0; i < lPhiToKK.size(); i++) {
    auto&& cand = lPhiToKK[i];

    auto closestPV = fillDecayHistograms(histos.phiToKK, cand, pvColl);
    if (not closestPV)
      continue;
    fillPhiToKKComponents(histos.phiToKK, cand, bs, closestPV);

    auto decayHistos = &histos.phiToKKPrompt;
    if (displacedPhiToKK[i]) {
      decayHistos = &histos.phiToKKDispl;
    }

    fillDecayHistograms(*decayHistos, cand, pvColl);
    fillPhiToKKComponents(*decayHistos, cand, bs, closestPV);
  }
  lPhiToKK.clear();
  displacedPhiToKK.clear();

  if (not bdToJPsiK0sCandsToken.isUninitialized()) {
    auto lBdToJPsiK0s = iEvent.get(bdToJPsiK0sCandsToken);
    for (auto&& cand : lBdToJPsiK0s) {
      auto jpsi = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi");
      auto jpsiMass = getMass(*jpsi);
      if (jpsiMass < 2.9 or jpsiMass > 3.3)
        continue;

      auto closestPV = fillDecayHistograms(histos.bdToJPsiK0s, cand, pvColl);
      if (not closestPV)
        continue;
      fillBdToJPsiK0sComponents(histos.bdToJPsiK0s, cand, bs, closestPV);

      displacedJPsiToMuMu[jpsi.index()] = true;
    }
  }

  if (not bcToJPsiPiCandsToken.isUninitialized()) {
    auto lBcToJPsiPi = iEvent.get(bcToJPsiPiCandsToken);
    for (auto&& cand : lBcToJPsiPi) {
      auto jpsi = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi");
      auto jpsiMass = getMass(*jpsi);
      if (jpsiMass < 2.9 or jpsiMass > 3.3)
        continue;

      auto closestPV = fillDecayHistograms(histos.bcToJPsiPi, cand, pvColl);
      if (not closestPV)
        continue;
      fillBcToJPsiPiComponents(histos.bcToJPsiPi, cand, bs, closestPV);

      displacedJPsiToMuMu[jpsi.index()] = true;
    }
  }

  if (not lambdaBToJPsiLambda0CandsToken.isUninitialized()) {
    auto lLambdaBToJPsiLambda0 = iEvent.get(lambdaBToJPsiLambda0CandsToken);
    for (auto&& cand : lLambdaBToJPsiLambda0) {
      auto jpsi = *cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi");
      auto jpsiMass = getMass(*jpsi);
      if (jpsiMass < 2.9 or jpsiMass > 3.3)
        continue;

      auto closestPV = fillDecayHistograms(histos.lambdaBToJPsiLambda0, cand, pvColl);
      if (not closestPV)
        continue;
      fillLambdaBToJPsiLambda0Components(histos.lambdaBToJPsiLambda0, cand, bs, closestPV);

      displacedJPsiToMuMu[jpsi.index()] = true;
    }
  }

  for (size_t i = 0; i < lOniaToMuMu.size(); i++) {
    auto&& cand = lOniaToMuMu[i];

    auto closestPV = fillDecayHistograms(histos.oniaToMuMu, cand, pvColl);
    if (not closestPV)
      continue;
    fillOniaToMuMuComponents(histos.oniaToMuMu, cand, bs, closestPV);

    auto decayHistos = &histos.oniaToMuMuPrompt;
    if (displacedJPsiToMuMu[i]) {
      decayHistos = &histos.oniaToMuMuDispl;
    }

    fillDecayHistograms(*decayHistos, cand, pvColl);
    fillOniaToMuMuComponents(*decayHistos, cand, bs, closestPV);
  }
  lOniaToMuMu.clear();
  displacedJPsiToMuMu.clear();

  if (not k0sToPiPiCandsToken.isUninitialized()) {
    auto lK0sToPiPi = iEvent.get(k0sToPiPiCandsToken);
    for (auto&& cand : lK0sToPiPi) {
      auto closestPV = fillDecayHistograms(histos.k0sToPiPi, cand, pvColl);
      if (not closestPV)
        continue;
      fillK0sToPiPiComponents(histos.k0sToPiPi, cand, bs, closestPV);
    }
  }

  if (not lambda0ToPPiCandsToken.isUninitialized()) {
    auto lLambda0ToPPi = iEvent.get(lambda0ToPPiCandsToken);
    for (auto&& cand : lLambda0ToPPi) {
      auto closestPV = fillDecayHistograms(histos.lambda0ToPPi, cand, pvColl);
      if (not closestPV)
        continue;
      fillLambda0ToPPiComponents(histos.lambda0ToPPi, cand, bs, closestPV);
    }
  }
}

void HeavyFlavorDQMAnalyzer::bookDecayHists(DQMStore::IBooker& ibook,
                                            edm::Run const&,
                                            edm::EventSetup const&,
                                            DecayHists& decayHists,
                                            std::string const& name,
                                            std::string const& products,
                                            int nMassBins,
                                            float massMin,
                                            float massMax,
                                            float distanceScaleFactor) const {
  std::string histTitle = name + " #rightarrow " + products + ";";

  decayHists.h_mass =
      ibook.book1D("h_mass", histTitle + "M(" + products + ") fitted [GeV]", nMassBins, massMin, massMax);
  decayHists.h_pt = ibook.book1D("h_pt", histTitle + "fitted p_{T} [GeV]", 100, 0.00, 200.0);
  decayHists.h_eta = ibook.book1D("h_eta", histTitle + "fitted #eta", 100, -3, 3);
  decayHists.h_phi = ibook.book1D("h_phi", histTitle + "fitted #varphi [rad]", 100, -TMath::Pi(), TMath::Pi());
  decayHists.h_displ2D =
      ibook.book1D("h_displ2D", histTitle + "vertex 2D displacement [cm]", 100, 0.00, 2.0 * distanceScaleFactor);
  decayHists.h_sign2D =
      ibook.book1D("h_sign2D", histTitle + "vertex 2D displ. significance", 100, 0.00, 200.0 * distanceScaleFactor);
  decayHists.h_ct = ibook.book1D("h_ct", histTitle + "ct [cm]", 100, 0.00, 0.4 * distanceScaleFactor);
  decayHists.h_pointing = ibook.book1D("h_pointing", histTitle + "cos( 2D pointing angle )", 100, -1, 1);
  decayHists.h_vertNormChi2 = ibook.book1D("h_vertNormChi2", histTitle + "vertex #chi^{2}/ndof", 100, 0.00, 10);
  decayHists.h_vertProb = ibook.book1D("h_vertProb", histTitle + "vertex prob.", 100, 0.00, 1.0);
}

void HeavyFlavorDQMAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                            edm::Run const& run,
                                            edm::EventSetup const& iSetup,
                                            Histograms& histos) const {
  if (not oniaToMuMuCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/JPsiToMuMuPrompt");
    bookDecayHists(ibook, run, iSetup, histos.oniaToMuMuPrompt, "J/#psi", "#mu^{+}#mu^{-}", 100, 2.9, 3.3);

    ibook.setCurrentFolder(folder_ + "/JPsiToMuMuPrompt/components");
    initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos.oniaToMuMuPrompt);

    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/JPsiToMuMuDisplaced");
    bookDecayHists(ibook, run, iSetup, histos.oniaToMuMuDispl, "J/#psi", "#mu^{+}#mu^{-}", 100, 2.9, 3.3);

    ibook.setCurrentFolder(folder_ + "/JPsiToMuMuDisplaced/components");
    initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos.oniaToMuMuDispl);

    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/OniaToMuMu");
    bookDecayHists(ibook, run, iSetup, histos.oniaToMuMu, "Onia", "#mu^{+}#mu^{-}", 750, 0, 15);

    ibook.setCurrentFolder(folder_ + "/OniaToMuMu/components");
    initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos.oniaToMuMu);
  }

  if (not kx0ToKPiCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/Kx0ToKPiPrompt");
    bookDecayHists(ibook, run, iSetup, histos.kx0ToKPiPrompt, "K*^{0}", "#pi^{+} K^{-}", 100, 0.75, 1.05);

    ibook.setCurrentFolder(folder_ + "/Kx0ToKPiPrompt/components");
    initKx0ToKPiComponentHistograms(ibook, run, iSetup, histos.kx0ToKPiPrompt);

    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/Kx0ToKPiDisplaced");
    bookDecayHists(ibook, run, iSetup, histos.kx0ToKPiDispl, "K*^{0}", "#pi^{+} K^{-}", 100, 0.75, 1.05);

    ibook.setCurrentFolder(folder_ + "/Kx0ToKPiDisplaced/components");
    initKx0ToKPiComponentHistograms(ibook, run, iSetup, histos.kx0ToKPiDispl);

    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/Kx0ToKPi");
    bookDecayHists(ibook, run, iSetup, histos.kx0ToKPi, "K*^{0}", "#pi^{+} K^{-}", 100, 0.75, 1.05);

    ibook.setCurrentFolder(folder_ + "/Kx0ToKPi/components");
    initKx0ToKPiComponentHistograms(ibook, run, iSetup, histos.kx0ToKPi);
  }

  if (not phiToKKCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/PhiToKKPrompt");
    bookDecayHists(ibook, run, iSetup, histos.phiToKKPrompt, "#phi", "K^{+} K^{-}", 100, 1.005, 1.035);

    ibook.setCurrentFolder(folder_ + "/PhiToKKPrompt/components");
    initPhiToKKComponentHistograms(ibook, run, iSetup, histos.phiToKKPrompt);

    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/PhiToKKDisplaced");
    bookDecayHists(ibook, run, iSetup, histos.phiToKKDispl, "#phi", "K^{+} K^{-}", 100, 1.005, 1.035);

    ibook.setCurrentFolder(folder_ + "/PhiToKKDisplaced/components");
    initPhiToKKComponentHistograms(ibook, run, iSetup, histos.phiToKKDispl);

    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/PhiToKK");
    bookDecayHists(ibook, run, iSetup, histos.phiToKK, "#phi", "K^{+} K^{-}", 100, 1.005, 1.035);

    ibook.setCurrentFolder(folder_ + "/PhiToKK/components");
    initPhiToKKComponentHistograms(ibook, run, iSetup, histos.phiToKK);
  }

  if (not psi2SToJPsiPiPiCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/Psi2SToJPsiPiPiPrompt");
    bookDecayHists(
        ibook, run, iSetup, histos.psi2SToJPsiPiPiPrompt, "#Psi(2S)", "J/#psi #pi^{+} #pi^{-}", 100, 3.65, 3.72);

    ibook.setCurrentFolder(folder_ + "/Psi2SToJPsiPiPiPrompt/components");
    initPsi2SToJPsiPiPiComponentHistograms(ibook, run, iSetup, histos.psi2SToJPsiPiPiPrompt);

    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/Psi2SToJPsiPiPiDisplaced");
    bookDecayHists(
        ibook, run, iSetup, histos.psi2SToJPsiPiPiDispl, "#Psi(2S)", "J/#psi #pi^{+} #pi^{-}", 100, 3.65, 3.72);

    ibook.setCurrentFolder(folder_ + "/Psi2SToJPsiPiPiDisplaced/components");
    initPsi2SToJPsiPiPiComponentHistograms(ibook, run, iSetup, histos.psi2SToJPsiPiPiDispl);

    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/Psi2SToJPsiPiPi");
    bookDecayHists(
        ibook, run, iSetup, histos.psi2SToJPsiPiPi, "#Psi(2S)/X(3872)", "J/#psi #pi^{+} #pi^{-}", 200, 3.60, 3.80);

    ibook.setCurrentFolder(folder_ + "/Psi2SToJPsiPiPi/components");
    initPsi2SToJPsiPiPiComponentHistograms(ibook, run, iSetup, histos.psi2SToJPsiPiPi);
  }

  if (not k0sToPiPiCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/K0sToPiPi");
    bookDecayHists(ibook, run, iSetup, histos.k0sToPiPi, "K^{0}_{S}", "#pi^{+} #pi^{-}", 100, 0.44, 0.56, 4);

    ibook.setCurrentFolder(folder_ + "/K0sToPiPi/components");
    initK0sToPiPiComponentHistograms(ibook, run, iSetup, histos.k0sToPiPi);
  }

  if (not lambda0ToPPiCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/Lambda0ToPPi");
    bookDecayHists(ibook, run, iSetup, histos.lambda0ToPPi, "#Lambda^{0}", "p^{+} #pi^{-}", 100, 1.06, 1.16, 4);

    ibook.setCurrentFolder(folder_ + "/Lambda0ToPPi/components");
    initLambda0ToPPiComponentHistograms(ibook, run, iSetup, histos.lambda0ToPPi);
  }

  if (not buToJPsiKCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/BuToJPsiK");
    bookDecayHists(ibook, run, iSetup, histos.buToJPsiK, "B^{+}", "J/#psi K^{+}", 100, 5.00, 6.00);

    ibook.setCurrentFolder(folder_ + "/BuToJPsiK/components");
    initBuToJPsiKComponentHistograms(ibook, run, iSetup, histos.buToJPsiK);
  }

  if (not buToPsi2SKCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/BuToPsi2SK");
    bookDecayHists(ibook, run, iSetup, histos.buToPsi2SK, "B^{+}", "#Psi(2S) K^{+}", 100, 5.00, 6.00);

    ibook.setCurrentFolder(folder_ + "/BuToPsi2SK/components");
    initBuToPsi2SKComponentHistograms(ibook, run, iSetup, histos.buToPsi2SK);
  }

  if (not bdToJPsiKx0CandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/BdToJPsiKx0");
    bookDecayHists(ibook, run, iSetup, histos.bdToJPsiKx0, "B^{0}", "J/#psi K*^{0}", 100, 5.00, 6.00);

    ibook.setCurrentFolder(folder_ + "/BdToJPsiKx0/components");
    initBdToJPsiKx0ComponentHistograms(ibook, run, iSetup, histos.bdToJPsiKx0);
  }

  if (not bsToJPsiPhiCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/BsToJPsiPhi");
    bookDecayHists(ibook, run, iSetup, histos.bsToJPsiPhi, "B^{0}_{s}", "J/#psi #phi", 100, 5.00, 6.00);

    ibook.setCurrentFolder(folder_ + "/BsToJPsiPhi/components");
    initBsToJPsiPhiComponentHistograms(ibook, run, iSetup, histos.bsToJPsiPhi);
  }

  if (not bdToJPsiK0sCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/BdToJPsiK0s");
    bookDecayHists(ibook, run, iSetup, histos.bdToJPsiK0s, "B^{0}", "J/#psi K^{0}_{S}", 100, 5.00, 6.00);

    ibook.setCurrentFolder(folder_ + "/BdToJPsiK0s/components");
    initBdToJPsiK0sComponentHistograms(ibook, run, iSetup, histos.bdToJPsiK0s);
  }

  if (not bcToJPsiPiCandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/BcToJPsiPi");
    bookDecayHists(ibook, run, iSetup, histos.bcToJPsiPi, "B^{+}_{c}", "J/#psi #pi^{+}", 100, 6.00, 7.00);

    ibook.setCurrentFolder(folder_ + "/BcToJPsiPi/components");
    initBcToJPsiPiComponentHistograms(ibook, run, iSetup, histos.bcToJPsiPi);
  }

  if (not lambdaBToJPsiLambda0CandsToken.isUninitialized()) {
    ibook.cd();
    ibook.setCurrentFolder(folder_ + "/LambdaBToJPsiLambda0");
    bookDecayHists(
        ibook, run, iSetup, histos.lambdaBToJPsiLambda0, "#Lambda^{0}_{b}", "J/#psi #Lambda^{0}", 100, 5.00, 6.00);

    ibook.setCurrentFolder(folder_ + "/LambdaBToJPsiLambda0/components");
    initLambdaBToJPsiLambda0ComponentHistograms(ibook, run, iSetup, histos.lambdaBToJPsiLambda0);
  }
}

void HeavyFlavorDQMAnalyzer::initComponentHists(DQMStore::IBooker& ibook,
                                                edm::Run const&,
                                                edm::EventSetup const&,
                                                DecayHists& histos,
                                                TString const& componentName) const {
  ComponentHists comp;

  comp.h_pt = ibook.book1D(componentName + "_pt", ";p_{T} [GeV]", 200, 0, 20);
  comp.h_eta = ibook.book1D(componentName + "_eta", ";#eta", 200, -3, 3);
  comp.h_phi = ibook.book1D(componentName + "_phi", ";#phi", 200, -TMath::Pi(), TMath::Pi());
  comp.h_dxy = ibook.book1D(componentName + "_dxyBS", ";d_{xy}(BS) [cm]", 200, -3, 3);
  comp.h_exy = ibook.book1D(componentName + "_exy", ";#sigma(d_{xy}(BS)) [cm]", 200, 0, 0.2);
  comp.h_dz = ibook.book1D(componentName + "_dzPV", ";d_{z}(PV) [cm]", 200, -20, 20);
  comp.h_ez = ibook.book1D(componentName + "_ez", ";#sigma(d_{z}(PV)) [cm]", 200, 0, 2);
  comp.h_chi2 = ibook.book1D(componentName + "_chi2", ";#chi^{2}", 200, 0, 20);

  histos.decayComponents.push_back(comp);
}

void HeavyFlavorDQMAnalyzer::initOniaToMuMuComponentHistograms(DQMStore::IBooker& ibook,
                                                               edm::Run const& run,
                                                               edm::EventSetup const& iSetup,
                                                               DecayHists& histos) const {
  initComponentHists(ibook, run, iSetup, histos, "lead_mu");
  initComponentHists(ibook, run, iSetup, histos, "soft_mu");
}

void HeavyFlavorDQMAnalyzer::initKx0ToKPiComponentHistograms(DQMStore::IBooker& ibook,
                                                             edm::Run const& run,
                                                             edm::EventSetup const& iSetup,
                                                             DecayHists& histos) const {
  initComponentHists(ibook, run, iSetup, histos, "k");
  initComponentHists(ibook, run, iSetup, histos, "pi");
}

void HeavyFlavorDQMAnalyzer::initPhiToKKComponentHistograms(DQMStore::IBooker& ibook,
                                                            edm::Run const& run,
                                                            edm::EventSetup const& iSetup,
                                                            DecayHists& histos) const {
  initComponentHists(ibook, run, iSetup, histos, "lead_k");
  initComponentHists(ibook, run, iSetup, histos, "soft_k");
}

void HeavyFlavorDQMAnalyzer::initK0sToPiPiComponentHistograms(DQMStore::IBooker& ibook,
                                                              edm::Run const& run,
                                                              edm::EventSetup const& iSetup,
                                                              DecayHists& histos) const {
  initComponentHists(ibook, run, iSetup, histos, "lead_pi");
  initComponentHists(ibook, run, iSetup, histos, "soft_pi");
}

void HeavyFlavorDQMAnalyzer::initLambda0ToPPiComponentHistograms(DQMStore::IBooker& ibook,
                                                                 edm::Run const& run,
                                                                 edm::EventSetup const& iSetup,
                                                                 DecayHists& histos) const {
  initComponentHists(ibook, run, iSetup, histos, "p");
  initComponentHists(ibook, run, iSetup, histos, "pi");
}

void HeavyFlavorDQMAnalyzer::initBuToJPsiKComponentHistograms(DQMStore::IBooker& ibook,
                                                              edm::Run const& run,
                                                              edm::EventSetup const& iSetup,
                                                              DecayHists& histos) const {
  initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos);
  initComponentHists(ibook, run, iSetup, histos, "k");
}

void HeavyFlavorDQMAnalyzer::initBuToPsi2SKComponentHistograms(DQMStore::IBooker& ibook,
                                                               edm::Run const& run,
                                                               edm::EventSetup const& iSetup,
                                                               DecayHists& histos) const {
  initPsi2SToJPsiPiPiComponentHistograms(ibook, run, iSetup, histos);
  initComponentHists(ibook, run, iSetup, histos, "k");
}

void HeavyFlavorDQMAnalyzer::initBdToJPsiKx0ComponentHistograms(DQMStore::IBooker& ibook,
                                                                edm::Run const& run,
                                                                edm::EventSetup const& iSetup,
                                                                DecayHists& histos) const {
  initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos);
  initKx0ToKPiComponentHistograms(ibook, run, iSetup, histos);
}

void HeavyFlavorDQMAnalyzer::initBsToJPsiPhiComponentHistograms(DQMStore::IBooker& ibook,
                                                                edm::Run const& run,
                                                                edm::EventSetup const& iSetup,
                                                                DecayHists& histos) const {
  initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos);
  initPhiToKKComponentHistograms(ibook, run, iSetup, histos);
}

void HeavyFlavorDQMAnalyzer::initBdToJPsiK0sComponentHistograms(DQMStore::IBooker& ibook,
                                                                edm::Run const& run,
                                                                edm::EventSetup const& iSetup,
                                                                DecayHists& histos) const {
  initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos);
  initK0sToPiPiComponentHistograms(ibook, run, iSetup, histos);
}

void HeavyFlavorDQMAnalyzer::initBcToJPsiPiComponentHistograms(DQMStore::IBooker& ibook,
                                                               edm::Run const& run,
                                                               edm::EventSetup const& iSetup,
                                                               DecayHists& histos) const {
  initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos);
  initComponentHists(ibook, run, iSetup, histos, "pi");
}

void HeavyFlavorDQMAnalyzer::initLambdaBToJPsiLambda0ComponentHistograms(DQMStore::IBooker& ibook,
                                                                         edm::Run const& run,
                                                                         edm::EventSetup const& iSetup,
                                                                         DecayHists& histos) const {
  initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos);
  initLambda0ToPPiComponentHistograms(ibook, run, iSetup, histos);
}

void HeavyFlavorDQMAnalyzer::initPsi2SToJPsiPiPiComponentHistograms(DQMStore::IBooker& ibook,
                                                                    edm::Run const& run,
                                                                    edm::EventSetup const& iSetup,
                                                                    DecayHists& histos) const {
  initOniaToMuMuComponentHistograms(ibook, run, iSetup, histos);
  initComponentHists(ibook, run, iSetup, histos, "lead_pi");
  initComponentHists(ibook, run, iSetup, histos, "soft_pi");
}

reco::Vertex const* HeavyFlavorDQMAnalyzer::fillDecayHistograms(DecayHists const& histos,
                                                                pat::CompositeCandidate const& cand,
                                                                reco::VertexCollection const& pvs) const {
  //  if (not cand.hasUserData("fitMomentum")) {
  //    return -2;
  //  }
  //  auto mass = cand.userFloat("fitMass");
  //  auto& momentum = *cand.userData<GlobalVector>("fitMomentum");
  if (not allTracksAvailable(cand)) {
    return nullptr;
  }

  auto svtx = cand.userData<reco::Vertex>("vertex");
  if (not svtx->isValid()) {
    return nullptr;
  }

  float mass = cand.mass();
  reco::Candidate::Vector momentum = cand.momentum();
  if (cand.hasUserData("fitMomentum")) {
    mass = cand.userFloat("fitMass");
    momentum = *cand.userData<GlobalVector>("fitMomentum");
  }

  auto pvtx = std::min_element(pvs.begin(), pvs.end(), [svtx](reco::Vertex const& pv1, reco::Vertex const& pv2) {
    return abs(pv1.z() - svtx->z()) < abs(pv2.z() - svtx->z());
  });
  if (pvtx == pvs.end()) {
    return nullptr;
  }

  VertexDistanceXY vdistXY;
  Measurement1D distXY = vdistXY.distance(*svtx, *pvtx);

  auto pvtPos = pvtx->position();
  auto svtPos = svtx->position();

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

  // FIXME workaround for tracks with non pos-def cov. matrix
  if (svtx->chi2() >= 0) {
    histos.h_vertNormChi2->Fill(svtx->chi2() / svtx->ndof());
    histos.h_vertProb->Fill(ChiSquaredProbability(svtx->chi2(), svtx->ndof()));
  }

  return &*pvtx;
}

int HeavyFlavorDQMAnalyzer::fillOniaToMuMuComponents(DecayHists const& histos,
                                                     pat::CompositeCandidate const& cand,
                                                     reco::BeamSpot const* bs,
                                                     reco::Vertex const* pv,
                                                     int startPosition) const {
  startPosition = fillComponentHistogramsLeadSoft(histos, cand, "MuPos", "MuNeg", bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillKx0ToKPiComponents(DecayHists const& histos,
                                                   pat::CompositeCandidate const& cand,
                                                   reco::BeamSpot const* bs,
                                                   reco::Vertex const* pv,
                                                   int startPosition) const {
  startPosition = fillComponentHistogramsSinglePart(histos, cand, "Kaon", bs, pv, startPosition);
  startPosition = fillComponentHistogramsSinglePart(histos, cand, "Pion", bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillPhiToKKComponents(DecayHists const& histos,
                                                  pat::CompositeCandidate const& cand,
                                                  reco::BeamSpot const* bs,
                                                  reco::Vertex const* pv,
                                                  int startPosition) const {
  startPosition = fillComponentHistogramsLeadSoft(histos, cand, "KPos", "KNeg", bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillK0sToPiPiComponents(DecayHists const& histos,
                                                    pat::CompositeCandidate const& cand,
                                                    reco::BeamSpot const* bs,
                                                    reco::Vertex const* pv,
                                                    int startPosition) const {
  startPosition = fillComponentHistogramsLeadSoft(histos, cand, "PionPos", "PionNeg", bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillLambda0ToPPiComponents(DecayHists const& histos,
                                                       pat::CompositeCandidate const& cand,
                                                       reco::BeamSpot const* bs,
                                                       reco::Vertex const* pv,
                                                       int startPosition) const {
  startPosition = fillComponentHistogramsSinglePart(histos, cand, "Proton", bs, pv, startPosition);
  startPosition = fillComponentHistogramsSinglePart(histos, cand, "Pion", bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillBuToJPsiKComponents(DecayHists const& histos,
                                                    pat::CompositeCandidate const& cand,
                                                    reco::BeamSpot const* bs,
                                                    reco::Vertex const* pv,
                                                    int startPosition) const {
  startPosition = fillOniaToMuMuComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi"), bs, pv, startPosition);
  startPosition = fillComponentHistogramsSinglePart(histos, cand, "Kaon", bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillBuToPsi2SKComponents(DecayHists const& histos,
                                                     pat::CompositeCandidate const& cand,
                                                     reco::BeamSpot const* bs,
                                                     reco::Vertex const* pv,
                                                     int startPosition) const {
  startPosition = fillPsi2SToJPsiPiPiComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToPsi2S"), bs, pv, startPosition);
  startPosition = fillComponentHistogramsSinglePart(histos, cand, "Kaon", bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillBdToJPsiKx0Components(DecayHists const& histos,
                                                      pat::CompositeCandidate const& cand,
                                                      reco::BeamSpot const* bs,
                                                      reco::Vertex const* pv,
                                                      int startPosition) const {
  startPosition = fillOniaToMuMuComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi"), bs, pv, startPosition);
  startPosition = fillKx0ToKPiComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToKx0"), bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillBsToJPsiPhiComponents(DecayHists const& histos,
                                                      pat::CompositeCandidate const& cand,
                                                      reco::BeamSpot const* bs,
                                                      reco::Vertex const* pv,
                                                      int startPosition) const {
  startPosition = fillOniaToMuMuComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi"), bs, pv, startPosition);
  startPosition = fillPhiToKKComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToPhi"), bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillBdToJPsiK0sComponents(DecayHists const& histos,
                                                      pat::CompositeCandidate const& cand,
                                                      reco::BeamSpot const* bs,
                                                      reco::Vertex const* pv,
                                                      int startPosition) const {
  startPosition = fillOniaToMuMuComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi"), bs, pv, startPosition);
  startPosition = fillK0sToPiPiComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToK0s"), bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillBcToJPsiPiComponents(DecayHists const& histos,
                                                     pat::CompositeCandidate const& cand,
                                                     reco::BeamSpot const* bs,
                                                     reco::Vertex const* pv,
                                                     int startPosition) const {
  startPosition = fillOniaToMuMuComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi"), bs, pv, startPosition);
  startPosition = fillComponentHistogramsSinglePart(histos, cand, "Pion", bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillLambdaBToJPsiLambda0Components(DecayHists const& histos,
                                                               pat::CompositeCandidate const& cand,
                                                               reco::BeamSpot const* bs,
                                                               reco::Vertex const* pv,
                                                               int startPosition) const {
  startPosition = fillOniaToMuMuComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi"), bs, pv, startPosition);
  startPosition = fillLambda0ToPPiComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToLambda0"), bs, pv, startPosition);

  return startPosition;
}

int HeavyFlavorDQMAnalyzer::fillPsi2SToJPsiPiPiComponents(DecayHists const& histos,
                                                          pat::CompositeCandidate const& cand,
                                                          reco::BeamSpot const* bs,
                                                          reco::Vertex const* pv,
                                                          int startPosition) const {
  startPosition = fillOniaToMuMuComponents(
      histos, **cand.userData<edm::Ref<pat::CompositeCandidateCollection>>("refToJPsi"), bs, pv, startPosition);
  startPosition = fillComponentHistogramsLeadSoft(histos, cand, "PionPos", "PionNeg", bs, pv, startPosition);
  return startPosition;
}

void HeavyFlavorDQMAnalyzer::fillComponentHistograms(ComponentHists const& histos,
                                                     reco::Track const& component,
                                                     reco::BeamSpot const* bs,
                                                     reco::Vertex const* pv) const {
  histos.h_pt->Fill(component.pt());
  histos.h_eta->Fill(component.eta());
  histos.h_phi->Fill(component.phi());

  math::XYZPoint zero(0, 0, 0);
  math::Error<3>::type zeroCov;  // needed for dxyError
  if (bs) {
    histos.h_dxy->Fill(component.dxy(*bs));
    histos.h_exy->Fill(component.dxyError(*bs));
  } else {
    histos.h_dxy->Fill(component.dxy(zero));
    histos.h_exy->Fill(component.dxyError(zero, zeroCov));
  }
  if (pv) {
    histos.h_dz->Fill(component.dz(pv->position()));
  } else {
    histos.h_dz->Fill(component.dz(zero));
  }
  histos.h_ez->Fill(component.dzError());

  histos.h_chi2->Fill(component.chi2() / component.ndof());
}

bool HeavyFlavorDQMAnalyzer::allTracksAvailable(pat::CompositeCandidate const& cand) const {
  for (auto&& name : cand.roles()) {
    auto track = getDaughterTrack(cand, name, false);
    if (not track) {
      return false;
    }
  }
  return true;
}

const reco::Track* HeavyFlavorDQMAnalyzer::getDaughterTrack(pat::CompositeCandidate const& cand,
                                                            std::string const& name,
                                                            bool throwOnMissing) const {
  auto daugh = cand.daughter(name);
  auto trackModeLabel = "trackMode_" + name;
  auto trackMode = cand.userData<std::string>(trackModeLabel);
  if (!trackMode or trackMode->empty()) {
    if (throwOnMissing) {
      throw cms::Exception("TrackNotFound") << "Could not determine track mode from candidate with name " << name
                                            << " with label " << trackModeLabel << std::endl;
    }
    return nullptr;
  }

  auto track = BPHTrackReference::getTrack(*daugh, trackMode->c_str());

  if (throwOnMissing and not track) {
    throw cms::Exception("TrackNotFound") << "BPHTrackReference could not extract a track as type " << trackMode
                                          << " from candidate with name " << name << std::endl;
  }

  return track;
}

int HeavyFlavorDQMAnalyzer::fillComponentHistogramsSinglePart(DecayHists const& histos,
                                                              pat::CompositeCandidate const& cand,
                                                              std::string const& name,
                                                              reco::BeamSpot const* bs,
                                                              reco::Vertex const* pv,
                                                              int startPosition) const {
  fillComponentHistograms(histos.decayComponents[startPosition], *getDaughterTrack(cand, name), bs, pv);

  return startPosition + 1;
}

int HeavyFlavorDQMAnalyzer::fillComponentHistogramsLeadSoft(DecayHists const& histos,
                                                            pat::CompositeCandidate const& cand,
                                                            std::string const& name1,
                                                            std::string const& name2,
                                                            reco::BeamSpot const* bs,
                                                            reco::Vertex const* pv,
                                                            int startPosition) const {
  auto daughSoft = getDaughterTrack(cand, name1);
  auto daughLead = getDaughterTrack(cand, name2);

  if (daughLead->pt() < daughSoft->pt()) {
    std::swap(daughLead, daughSoft);
  }

  fillComponentHistograms(histos.decayComponents[startPosition], *daughLead, bs, pv);
  fillComponentHistograms(histos.decayComponents[startPosition + 1], *daughSoft, bs, pv);

  return startPosition + 2;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HeavyFlavorDQMAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "Physics/HeavyFlavor");

  desc.add<edm::InputTag>("pvCollection");
  desc.add<edm::InputTag>("beamSpot");

  desc.addOptional<edm::InputTag>("OniaToMuMuCands");
  desc.addOptional<edm::InputTag>("Kx0ToKPiCands");
  desc.addOptional<edm::InputTag>("PhiToKKCands");
  desc.addOptional<edm::InputTag>("BuToJPsiKCands");
  desc.addOptional<edm::InputTag>("BuToPsi2SKCands");
  desc.addOptional<edm::InputTag>("BdToJPsiKx0Cands");
  desc.addOptional<edm::InputTag>("BsToJPsiPhiCands");
  desc.addOptional<edm::InputTag>("K0sToPiPiCands");
  desc.addOptional<edm::InputTag>("Lambda0ToPPiCands");
  desc.addOptional<edm::InputTag>("BdToJPsiK0sCands");
  desc.addOptional<edm::InputTag>("LambdaBToJPsiLambda0Cands");
  desc.addOptional<edm::InputTag>("BcToJPsiPiCands");
  desc.addOptional<edm::InputTag>("Psi2SToJPsiPiPiCands");

  descriptions.add("HeavyFlavorDQMAnalyzer", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HeavyFlavorDQMAnalyzer);
