#include "DQM/L1TMonitor/interface/L1TStage2MuonComp.h"


L1TStage2MuonComp::L1TStage2MuonComp(const edm::ParameterSet& ps)
    : muonToken1(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonCollection1"))),
      muonToken2(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonCollection2"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      muonColl1Title(ps.getUntrackedParameter<std::string>("muonCollection1Title", "Muon collection 1")),
      muonColl2Title(ps.getUntrackedParameter<std::string>("muonCollection2Title", "Muon collection 2")),
      summaryTitle(ps.getUntrackedParameter<std::string>("summaryTitle", "Summary")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false))
{
}

L1TStage2MuonComp::~L1TStage2MuonComp() {}

void L1TStage2MuonComp::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2MuonComp::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2MuonComp::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  // Subsystem Monitoring and Muon Output
  ibooker.setCurrentFolder(monitorDir);

  summary = ibooker.book1D("summary", summaryTitle.c_str(), 14, 1, 15); // range to match bin numbering
  summary->setBinLabel(BXRANGEGOOD, "BX range match", 1);
  summary->setBinLabel(BXRANGEBAD, "BX range mismatch", 1);
  summary->setBinLabel(NMUONGOOD, "muon collection size match", 1);
  summary->setBinLabel(NMUONBAD, "muon collection size mismatch", 1);
  summary->setBinLabel(MUONALL, "# muons", 1);
  summary->setBinLabel(MUONGOOD, "# matching muons", 1);
  summary->setBinLabel(PTBAD, "p_{T} mismatch", 1);
  summary->setBinLabel(ETABAD, "#eta mismatch", 1);
  summary->setBinLabel(PHIBAD, "#phi mismatch", 1);
  summary->setBinLabel(CHARGEBAD, "charge mismatch", 1);
  summary->setBinLabel(CHARGEVALBAD, "charge valid mismatch", 1);
  summary->setBinLabel(QUALBAD, "quality mismatch", 1);
  summary->setBinLabel(ISOBAD, "iso mismatch", 1);
  summary->setBinLabel(IDXBAD, "index mismatch", 1);

  muColl1BxRange = ibooker.book1D("muColl1BxRange", (muonColl1Title+" mismatching BX range").c_str(), 5, -2.5, 2.5);
  muColl1BxRange->setAxisTitle("BX range", 1);
  muColl1nMu = ibooker.book1D("muColl1nMu", (muonColl1Title+" mismatching muon multiplicity").c_str(), 9, -0.5, 8.5);
  muColl1nMu->setAxisTitle("Muon multiplicity", 1);
  muColl1hwPt = ibooker.book1D("muColl1hwPt", (muonColl1Title+" mismatching muon p_{T}").c_str(), 512, -0.5, 511.5);
  muColl1hwPt->setAxisTitle("Hardware p_{T}", 1);
  muColl1hwEta = ibooker.book1D("muColl1hwEta", (muonColl1Title+" mismatching muon #eta").c_str(), 461, -230.5, 230.5);
  muColl1hwEta->setAxisTitle("Hardware #eta", 1);
  muColl1hwPhi = ibooker.book1D("muColl1hwPhi", (muonColl1Title+" mismatching muon #phi").c_str(), 576, -0.5, 575.5);
  muColl1hwPhi->setAxisTitle("Hardware #phi", 1);
  muColl1hwCharge = ibooker.book1D("muColl1hwCharge", (muonColl1Title+" mismatching muon charge").c_str(), 2, -0.5, 1.5);
  muColl1hwCharge->setAxisTitle("Hardware charge", 1);
  muColl1hwChargeValid = ibooker.book1D("muColl1hwChargeValid", (muonColl1Title+" mismatching muon charge valid").c_str(), 2, -0.5, 1.5);
  muColl1hwChargeValid->setAxisTitle("Hardware charge valid", 1);
  muColl1hwQual = ibooker.book1D("muColl1hwQual", (muonColl1Title+" mismatching muon quality").c_str(), 16, -0.5, 15.5);
  muColl1hwQual->setAxisTitle("Hardware quality", 1);
  muColl1hwIso = ibooker.book1D("muColl1hwIso", (muonColl1Title+" mismatching muon isolation").c_str(), 4, -0.5, 3.5);
  muColl1hwIso->setAxisTitle("Hardware isolation", 1);
  muColl1Index = ibooker.book1D("muColl1Index", (muonColl1Title+" mismatching Input muon index").c_str(), 108, -0.5, 107.5);
  muColl1Index->setAxisTitle("Index", 1);

  muColl2BxRange = ibooker.book1D("muColl2BxRange", (muonColl2Title+" mismatching BX range").c_str(), 5, -2.5, 2.5);
  muColl2BxRange->setAxisTitle("BX range", 1);
  muColl2nMu = ibooker.book1D("muColl2nMu", (muonColl2Title+" mismatching muon multiplicity").c_str(), 9, -0.5, 8.5);
  muColl2nMu->setAxisTitle("Muon multiplicity", 1);
  muColl2hwPt = ibooker.book1D("muColl2hwPt", (muonColl2Title+" mismatching muon p_{T}").c_str(), 512, -0.5, 511.5);
  muColl2hwPt->setAxisTitle("Hardware p_{T}", 1);
  muColl2hwEta = ibooker.book1D("muColl2hwEta", (muonColl2Title+" mismatching muon #eta").c_str(), 461, -230.5, 230.5);
  muColl2hwEta->setAxisTitle("Hardware #eta", 1);
  muColl2hwPhi = ibooker.book1D("muColl2hwPhi", (muonColl2Title+" mismatching muon #phi").c_str(), 576, -0.5, 575.5);
  muColl2hwPhi->setAxisTitle("Hardware #phi", 1);
  muColl2hwCharge = ibooker.book1D("muColl2hwCharge", (muonColl2Title+" mismatching muon charge").c_str(), 2, -0.5, 1.5);
  muColl2hwCharge->setAxisTitle("Hardware charge", 1);
  muColl2hwChargeValid = ibooker.book1D("muColl2hwChargeValid", (muonColl2Title+" mismatching muon charge valid").c_str(), 2, -0.5, 1.5);
  muColl2hwChargeValid->setAxisTitle("Hardware charge valid", 1);
  muColl2hwQual = ibooker.book1D("muColl2hwQual", (muonColl2Title+" mismatching muon quality").c_str(), 16, -0.5, 15.5);
  muColl2hwQual->setAxisTitle("Hardware quality", 1);
  muColl2hwIso = ibooker.book1D("muColl2hwIso", (muonColl2Title+" mismatching muon isolation").c_str(), 4, -0.5, 3.5);
  muColl2hwIso->setAxisTitle("Hardware isolation", 1);
  muColl2Index = ibooker.book1D("muColl2Index", (muonColl2Title+" mismatching Input muon index").c_str(), 108, -0.5, 107.5);
  muColl2Index->setAxisTitle("Index", 1);
}

void L1TStage2MuonComp::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2MuonComp") << "L1TStage2MuonComp: analyze..." << std::endl;

  edm::Handle<l1t::MuonBxCollection> muonBxColl1;
  edm::Handle<l1t::MuonBxCollection> muonBxColl2;
  e.getByToken(muonToken1, muonBxColl1);
  e.getByToken(muonToken2, muonBxColl2);

  int bxRange1 = muonBxColl1->getLastBX() - muonBxColl1->getFirstBX() + 1;
  int bxRange2 = muonBxColl2->getLastBX() - muonBxColl2->getFirstBX() + 1;
  if (bxRange1 != bxRange2) {
    summary->Fill(BXRANGEBAD);
    int bx;
    for (bx = muonBxColl1->getFirstBX(); bx <= muonBxColl1->getLastBX(); ++bx) {
        muColl1BxRange->Fill(bx);
    }
    for (bx = muonBxColl2->getFirstBX(); bx <= muonBxColl2->getLastBX(); ++bx) {
        muColl2BxRange->Fill(bx);
    }
  } else {
    summary->Fill(BXRANGEGOOD);
  }

  for (int iBx = muonBxColl1->getFirstBX(); iBx <= muonBxColl1->getLastBX(); ++iBx) {
    // don't analyse if this BX does not exist in the second collection
    if (iBx < muonBxColl2->getFirstBX() || iBx > muonBxColl2->getLastBX()) continue;

    l1t::MuonBxCollection::const_iterator muonIt1;
    l1t::MuonBxCollection::const_iterator muonIt2;

    // check number of muons
    if (muonBxColl1->size(iBx) != muonBxColl2->size(iBx)) {
      summary->Fill(NMUONBAD);
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
    } else {
      summary->Fill(NMUONGOOD);
    }

    muonIt1 = muonBxColl1->begin(iBx);
    muonIt2 = muonBxColl2->begin(iBx);
    while(muonIt1 != muonBxColl1->end(iBx) && muonIt2 != muonBxColl2->end(iBx)) {
      summary->Fill(MUONALL);

      bool muonMismatch = false;
      if (muonIt1->hwPt() != muonIt2->hwPt()) {
        muonMismatch = true;
        summary->Fill(PTBAD);
      }
      if (muonIt1->hwEta() != muonIt2->hwEta()) {
        muonMismatch = true;
        summary->Fill(ETABAD);
      }
      if (muonIt1->hwPhi() != muonIt2->hwPhi()) {
        muonMismatch = true;
        summary->Fill(PHIBAD);
      }
      if (muonIt1->hwCharge() != muonIt2->hwCharge()) {
        muonMismatch = true;
        summary->Fill(CHARGEBAD);
      }
      if (muonIt1->hwChargeValid() != muonIt2->hwChargeValid()) {
        muonMismatch = true;
        summary->Fill(CHARGEVALBAD);
      }
      if (muonIt1->hwQual() != muonIt2->hwQual()) {
        muonMismatch = true;
        summary->Fill(QUALBAD);
      }
      if (muonIt1->hwIso() != muonIt2->hwIso()) {
        muonMismatch = true;
        summary->Fill(ISOBAD);
      }
      if (muonIt1->tfMuonIndex() != muonIt2->tfMuonIndex()) {
        muonMismatch = true;
        summary->Fill(IDXBAD);
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
      } else {
        summary->Fill(MUONGOOD);
      }

      ++muonIt1;
      ++muonIt2;
    }
  }
}

