#include "DQM/L1TMonitor/interface/L1TStage2MuonComp.h"


L1TStage2MuonComp::L1TStage2MuonComp(const edm::ParameterSet& ps)
    : muonToken1(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonCollection1"))),
      muonToken2(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonCollection2"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      muonColl1Title(ps.getUntrackedParameter<std::string>("muonCollection1Title", "Muon collection 1")),
      muonColl2Title(ps.getUntrackedParameter<std::string>("muonCollection2Title", "Muon collection 2")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false))
{
}

L1TStage2MuonComp::~L1TStage2MuonComp() {}

void L1TStage2MuonComp::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2MuonComp::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2MuonComp::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  // Subsystem Monitoring and Muon Output
  ibooker.setCurrentFolder(monitorDir);

  mismatchSummary = ibooker.book1D("mismatchSummary", "Mismatch summary", 10, 0, 10);
  mismatchSummary->setBinLabel(BXRANGE, "BX range", 1);
  mismatchSummary->setBinLabel(NMUON, "# muons", 1);
  mismatchSummary->setBinLabel(PT, "p_{T}", 1);
  mismatchSummary->setBinLabel(ETA, "#eta", 1);
  mismatchSummary->setBinLabel(PHI, "#phi", 1);
  mismatchSummary->setBinLabel(CHARGE, "charge", 1);
  mismatchSummary->setBinLabel(CHARGEVAL, "charge valid", 1);
  mismatchSummary->setBinLabel(QUAL, "quality", 1);
  mismatchSummary->setBinLabel(ISO, "iso", 1);
  mismatchSummary->setBinLabel(IDX, "index", 1);

  muColl1BxRange = ibooker.book1D("muColl1BxRange", (muonColl1Title+" BX range").c_str(), 5, -2.5, 2.5);
  muColl1BxRange->setAxisTitle("BX range", 1);
  muColl1nMu = ibooker.book1D("muColl1nMu", (muonColl1Title+" muon multiplicity").c_str(), 9, -0.5, 8.5);
  muColl1nMu->setAxisTitle("Muon multiplicity (BX == 0)", 1);
  muColl1hwPt = ibooker.book1D("muColl1hwPt", (muonColl1Title+" muon p_{T}").c_str(), 512, -0.5, 511.5);
  muColl1hwPt->setAxisTitle("Hardware p_{T}", 1);
  muColl1hwEta = ibooker.book1D("muColl1hwEta", (muonColl1Title+" muon #eta").c_str(), 461, -230.5, 230.5);
  muColl1hwEta->setAxisTitle("Hardware #eta", 1);
  muColl1hwPhi = ibooker.book1D("muColl1hwPhi", (muonColl1Title+" muon #phi").c_str(), 576, -0.5, 575.5);
  muColl1hwPhi->setAxisTitle("Hardware #phi", 1);
  muColl1hwCharge = ibooker.book1D("muColl1hwCharge", (muonColl1Title+" muon charge").c_str(), 2, -0.5, 1.5);
  muColl1hwCharge->setAxisTitle("Hardware charge", 1);
  muColl1hwChargeValid = ibooker.book1D("muColl1hwChargeValid", (muonColl1Title+" muon charge valid").c_str(), 2, -0.5, 1.5);
  muColl1hwChargeValid->setAxisTitle("Hardware charge valid", 1);
  muColl1hwQual = ibooker.book1D("muColl1hwQual", (muonColl1Title+" muon quality").c_str(), 16, -0.5, 15.5);
  muColl1hwQual->setAxisTitle("Hardware quality", 1);
  muColl1hwIso = ibooker.book1D("muColl1hwIso", (muonColl1Title+" muon isolation").c_str(), 4, -0.5, 3.5);
  muColl1hwIso->setAxisTitle("Hardware isolation", 1);
  muColl1Index = ibooker.book1D("muColl1Index", (muonColl1Title+" Input muon index").c_str(), 108, -0.5, 107.5);
  muColl1Index->setAxisTitle("Index", 1);

  muColl2BxRange = ibooker.book1D("muColl2BxRange", (muonColl2Title+" BX range").c_str(), 5, -2.5, 2.5);
  muColl2BxRange->setAxisTitle("BX range", 1);
  muColl2nMu = ibooker.book1D("muColl2nMu", (muonColl2Title+" muon multiplicity").c_str(), 9, -0.5, 8.5);
  muColl2nMu->setAxisTitle("Muon multiplicity (BX == 0)", 1);
  muColl2hwPt = ibooker.book1D("muColl2hwPt", (muonColl2Title+" muon p_{T}").c_str(), 512, -0.5, 511.5);
  muColl2hwPt->setAxisTitle("Hardware p_{T}", 1);
  muColl2hwEta = ibooker.book1D("muColl2hwEta", (muonColl2Title+" muon #eta").c_str(), 461, -230.5, 230.5);
  muColl2hwEta->setAxisTitle("Hardware #eta", 1);
  muColl2hwPhi = ibooker.book1D("muColl2hwPhi", (muonColl2Title+" muon #phi").c_str(), 576, -0.5, 575.5);
  muColl2hwPhi->setAxisTitle("Hardware #phi", 1);
  muColl2hwCharge = ibooker.book1D("muColl2hwCharge", (muonColl2Title+" muon charge").c_str(), 2, -0.5, 1.5);
  muColl2hwCharge->setAxisTitle("Hardware charge", 1);
  muColl2hwChargeValid = ibooker.book1D("muColl2hwChargeValid", (muonColl2Title+" muon charge valid").c_str(), 2, -0.5, 1.5);
  muColl2hwChargeValid->setAxisTitle("Hardware charge valid", 1);
  muColl2hwQual = ibooker.book1D("muColl2hwQual", (muonColl2Title+" muon quality").c_str(), 16, -0.5, 15.5);
  muColl2hwQual->setAxisTitle("Hardware quality", 1);
  muColl2hwIso = ibooker.book1D("muColl2hwIso", (muonColl2Title+" muon isolation").c_str(), 4, -0.5, 3.5);
  muColl2hwIso->setAxisTitle("Hardware isolation", 1);
  muColl2Index = ibooker.book1D("muColl2Index", (muonColl2Title+" Input muon index").c_str(), 108, -0.5, 107.5);
  muColl2Index->setAxisTitle("Index", 1);
}

void L1TStage2MuonComp::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2MuonComp") << "L1TStage2MuonComp: analyze..." << std::endl;

  edm::Handle<l1t::MuonBxCollection> muonBxColl1;
  edm::Handle<l1t::MuonBxCollection> muonBxColl2;
  e.getByToken(muonToken1, muonBxColl1);
  e.getByToken(muonToken2, muonBxColl2);

  bool muonMismatch = false;

  int bxRange1 = muonBxColl1->getLastBX() - muonBxColl1->getFirstBX();
  int bxRange2 = muonBxColl2->getLastBX() - muonBxColl2->getFirstBX();
  if (bxRange1 != bxRange1) {
    mismatchSummary->Fill(BXRANGE);
    muColl1BxRange->Fill(bxRange1);
    muColl2BxRange->Fill(bxRange2);
  }

  for (int iBx = muonBxColl1->getFirstBX(); iBx <= muonBxColl1->getLastBX(); ++iBx) {
    // don't analyse if this BX does not exist in the second collection
    if (iBx < muonBxColl2->getFirstBX() || iBx > muonBxColl2->getLastBX()) continue;

    l1t::MuonBxCollection::const_iterator muonIt1;
    l1t::MuonBxCollection::const_iterator muonIt2;

    // check number of muons
    if (muonBxColl1->size(iBx) != muonBxColl2->size(iBx)) {
      mismatchSummary->Fill(NMUON);
      muColl1nMu->Fill(muonBxColl1->size(iBx));
      muColl2nMu->Fill(muonBxColl2->size(iBx));

      if (muonBxColl1->size(iBx) > muonBxColl2->size(iBx)) {
        muonIt1 = muonBxColl1->begin(iBx) + muonBxColl2->size(iBx);
        for (; muonIt1 != muonBxColl1->end(iBx); ++muonIt1) {
          muColl1hwPt->Fill(muonIt1->hwPt());
          muColl1hwEta->Fill(muonIt1->hwEta());
          muColl1hwPhi->Fill(muonIt1->hwPhi());
          muColl1hwCharge->Fill(muonIt1->hwCharge());
          muColl1hwChargeValid->Fill(muonIt1->hwChargeValid());
          muColl1hwQual->Fill(muonIt1->hwQual());
          muColl1hwIso->Fill(muonIt1->hwIso());
          muColl1Index->Fill(muonIt1->tfMuonIndex());
        }
      } else {
        muonIt2 = muonBxColl2->begin(iBx) + muonBxColl1->size(iBx);
        for (; muonIt2 != muonBxColl2->end(iBx); ++muonIt2) {
          muColl2hwPt->Fill(muonIt2->hwPt());
          muColl2hwEta->Fill(muonIt2->hwEta());
          muColl2hwPhi->Fill(muonIt2->hwPhi());
          muColl2hwCharge->Fill(muonIt2->hwCharge());
          muColl2hwChargeValid->Fill(muonIt2->hwChargeValid());
          muColl2hwQual->Fill(muonIt2->hwQual());
          muColl2hwIso->Fill(muonIt2->hwIso());
          muColl2Index->Fill(muonIt2->tfMuonIndex());
        }
      }
    }

    muonIt1 = muonBxColl1->begin(iBx);
    muonIt2 = muonBxColl2->begin(iBx);
    while(muonIt1 != muonBxColl1->end(iBx) && muonIt2 != muonBxColl1->end(iBx)) {
      if (muonIt1->hwPt() != muonIt2->hwPt()) {
        muonMismatch = true;
        mismatchSummary->Fill(PT);
      }
      if (muonIt1->hwEta() != muonIt2->hwEta()) {
        muonMismatch = true;
        mismatchSummary->Fill(ETA);
      }
      if (muonIt1->hwPhi() != muonIt2->hwPhi()) {
        muonMismatch = true;
        mismatchSummary->Fill(PHI);
      }
      if (muonIt1->hwCharge() != muonIt2->hwCharge()) {
        muonMismatch = true;
        mismatchSummary->Fill(CHARGE);
      }
      if (muonIt1->hwChargeValid() != muonIt2->hwChargeValid()) {
        muonMismatch = true;
        mismatchSummary->Fill(CHARGEVAL);
      }
      if (muonIt1->hwQual() != muonIt2->hwQual()) {
        muonMismatch = true;
        mismatchSummary->Fill(QUAL);
      }
      if (muonIt1->hwIso() != muonIt2->hwIso()) {
        muonMismatch = true;
        mismatchSummary->Fill(ISO);
      }
      if (muonIt1->tfMuonIndex() != muonIt2->tfMuonIndex()) {
        muonMismatch = true;
        mismatchSummary->Fill(IDX);
      }

      if (muonMismatch) {
        muColl1hwPt->Fill(muonIt1->hwPt());
        muColl1hwEta->Fill(muonIt1->hwEta());
        muColl1hwPhi->Fill(muonIt1->hwPhi());
        muColl1hwCharge->Fill(muonIt1->hwCharge());
        muColl1hwChargeValid->Fill(muonIt1->hwChargeValid());
        muColl1hwQual->Fill(muonIt1->hwQual());
        muColl1hwIso->Fill(muonIt1->hwIso());
        muColl1Index->Fill(muonIt1->tfMuonIndex());

        muColl2hwPt->Fill(muonIt2->hwPt());
        muColl2hwEta->Fill(muonIt2->hwEta());
        muColl2hwPhi->Fill(muonIt2->hwPhi());
        muColl2hwCharge->Fill(muonIt2->hwCharge());
        muColl2hwChargeValid->Fill(muonIt2->hwChargeValid());
        muColl2hwQual->Fill(muonIt2->hwQual());
        muColl2hwIso->Fill(muonIt2->hwIso());
        muColl2Index->Fill(muonIt2->tfMuonIndex());
      }

      ++muonIt1;
      ++muonIt2;
    }
  }
}

