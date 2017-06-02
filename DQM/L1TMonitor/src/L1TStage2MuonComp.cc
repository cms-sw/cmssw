#include "DQM/L1TMonitor/interface/L1TStage2MuonComp.h"


L1TStage2MuonComp::L1TStage2MuonComp(const edm::ParameterSet& ps)
    : muonToken1(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonCollection1"))),
      muonToken2(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonCollection2"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir")),
      muonColl1Title(ps.getUntrackedParameter<std::string>("muonCollection1Title")),
      muonColl2Title(ps.getUntrackedParameter<std::string>("muonCollection2Title")),
      summaryTitle(ps.getUntrackedParameter<std::string>("summaryTitle")),
      verbose(ps.getUntrackedParameter<bool>("verbose"))
{
}

L1TStage2MuonComp::~L1TStage2MuonComp() {}

void L1TStage2MuonComp::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonCollection1")->setComment("L1T Muon collection 1");
  desc.add<edm::InputTag>("muonCollection2")->setComment("L1T Muon collection 2");
  desc.addUntracked<std::string>("monitorDir", "")->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<std::string>("muonCollection1Title", "Muon collection 1")->setComment("Histogram title for first collection.");
  desc.addUntracked<std::string>("muonCollection2Title", "Muon collection 2")->setComment("Histogram title for second collection.");
  desc.addUntracked<std::string>("summaryTitle", "Summary")->setComment("Title of summary histogram.");
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("l1tStage2MuonComp", desc);
}

void L1TStage2MuonComp::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2MuonComp::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2MuonComp::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  // Subsystem Monitoring and Muon Output
  ibooker.setCurrentFolder(monitorDir);

  summary = ibooker.book1D("summary", summaryTitle.c_str(), 16, 1, 17); // range to match bin numbering
  summary->setBinLabel(BXRANGEGOOD, "BX range match", 1);
  summary->setBinLabel(BXRANGEBAD, "BX range mismatch", 1);
  summary->setBinLabel(NMUONGOOD, "muon collection size match", 1);
  summary->setBinLabel(NMUONBAD, "muon collection size mismatch", 1);
  summary->setBinLabel(MUONALL, "# muons", 1);
  summary->setBinLabel(MUONGOOD, "# matching muons", 1);
  summary->setBinLabel(PTBAD, "p_{T} mismatch", 1);
  summary->setBinLabel(ETABAD, "#eta mismatch", 1);
  summary->setBinLabel(PHIBAD, "#phi mismatch", 1);
  summary->setBinLabel(ETAATVTXBAD, "#eta at vertex mismatch", 1);
  summary->setBinLabel(PHIATVTXBAD, "#phi at vertex mismatch", 1);
  summary->setBinLabel(CHARGEBAD, "charge mismatch", 1);
  summary->setBinLabel(CHARGEVALBAD, "charge valid mismatch", 1);
  summary->setBinLabel(QUALBAD, "quality mismatch", 1);
  summary->setBinLabel(ISOBAD, "iso mismatch", 1);
  summary->setBinLabel(IDXBAD, "index mismatch", 1);

  errorSummaryNum = ibooker.book1D("errorSummaryNum", summaryTitle.c_str(), 13, 1, 14); // range to match bin numbering
  errorSummaryNum->setBinLabel(RBXRANGE, "BX range mismatch", 1);
  errorSummaryNum->setBinLabel(RNMUON, "muon collection size mismatch", 1);
  errorSummaryNum->setBinLabel(RMUON, "mismatching muons", 1);
  errorSummaryNum->setBinLabel(RPT, "p_{T} mismatch", 1);
  errorSummaryNum->setBinLabel(RETA, "#eta mismatch", 1);
  errorSummaryNum->setBinLabel(RPHI, "#phi mismatch", 1);
  errorSummaryNum->setBinLabel(RETAATVTX, "#eta at vertex mismatch", 1);
  errorSummaryNum->setBinLabel(RPHIATVTX, "#phi at vertex mismatch", 1);
  errorSummaryNum->setBinLabel(RCHARGE, "charge mismatch", 1);
  errorSummaryNum->setBinLabel(RCHARGEVAL, "charge valid mismatch", 1);
  errorSummaryNum->setBinLabel(RQUAL, "quality mismatch", 1);
  errorSummaryNum->setBinLabel(RISO, "iso mismatch", 1);
  errorSummaryNum->setBinLabel(RIDX, "index mismatch", 1);

  errorSummaryDen = ibooker.book1D("errorSummaryDen", "denominators", 13, 1, 14); // range to match bin numbering
  errorSummaryDen->setBinLabel(RBXRANGE, "# events", 1);
  errorSummaryDen->setBinLabel(RNMUON, "# muon collections", 1);
  errorSummaryDen->setBinLabel(RMUON, "# muons", 1);
  errorSummaryDen->setBinLabel(RPT, "# muons", 1);
  errorSummaryDen->setBinLabel(RETA, "# muons", 1);
  errorSummaryDen->setBinLabel(RPHI, "# muons", 1);
  errorSummaryDen->setBinLabel(RETAATVTX, "# muons", 1);
  errorSummaryDen->setBinLabel(RPHIATVTX, "# muons", 1);
  errorSummaryDen->setBinLabel(RCHARGE, "# muons", 1);
  errorSummaryDen->setBinLabel(RCHARGEVAL, "# muons", 1);
  errorSummaryDen->setBinLabel(RQUAL, "# muons", 1);
  errorSummaryDen->setBinLabel(RISO, "# muons", 1);
  errorSummaryDen->setBinLabel(RIDX, "# muons", 1);

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
  muColl1hwEtaAtVtx = ibooker.book1D("muColl1hwEtaAtVtx", (muonColl1Title+" mismatching muon #eta at vertex").c_str(), 461, -230.5, 230.5);
  muColl1hwEtaAtVtx->setAxisTitle("Hardware #eta at vertex", 1);
  muColl1hwPhiAtVtx = ibooker.book1D("muColl1hwPhiAtVtx", (muonColl1Title+" mismatching muon #phi at vertex").c_str(), 576, -0.5, 575.5);
  muColl1hwPhiAtVtx->setAxisTitle("Hardware #phi at vertex", 1);
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
  muColl2hwEtaAtVtx = ibooker.book1D("muColl2hwEtaAtVtx", (muonColl2Title+" mismatching muon #eta at vertex").c_str(), 461, -230.5, 230.5);
  muColl2hwEtaAtVtx->setAxisTitle("Hardware #eta at vertex", 1);
  muColl2hwPhiAtVtx = ibooker.book1D("muColl2hwPhiAtVtx", (muonColl2Title+" mismatching muon #phi at vertex").c_str(), 576, -0.5, 575.5);
  muColl2hwPhiAtVtx->setAxisTitle("Hardware #phi at vertex", 1);
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

  errorSummaryDen->Fill(RBXRANGE);
  int bxRange1 = muonBxColl1->getLastBX() - muonBxColl1->getFirstBX() + 1;
  int bxRange2 = muonBxColl2->getLastBX() - muonBxColl2->getFirstBX() + 1;
  if (bxRange1 != bxRange2) {
    summary->Fill(BXRANGEBAD);
    errorSummaryNum->Fill(RBXRANGE);
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

    errorSummaryDen->Fill(RNMUON);
    // check number of muons
    if (muonBxColl1->size(iBx) != muonBxColl2->size(iBx)) {
      summary->Fill(NMUONBAD);
      errorSummaryNum->Fill(RNMUON);
      muColl1nMu->Fill(muonBxColl1->size(iBx));
      muColl2nMu->Fill(muonBxColl2->size(iBx));

      if (muonBxColl1->size(iBx) > muonBxColl2->size(iBx)) {
        muonIt1 = muonBxColl1->begin(iBx) + muonBxColl2->size(iBx);
        for (; muonIt1 != muonBxColl1->end(iBx); ++muonIt1) {
          muColl1hwPt->Fill(muonIt1->hwPt());
          muColl1hwEta->Fill(muonIt1->hwEta());
          muColl1hwPhi->Fill(muonIt1->hwPhi());
          muColl1hwEtaAtVtx->Fill(muonIt1->hwEtaAtVtx());
          muColl1hwPhiAtVtx->Fill(muonIt1->hwPhiAtVtx());
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
          muColl2hwEtaAtVtx->Fill(muonIt2->hwEtaAtVtx());
          muColl2hwPhiAtVtx->Fill(muonIt2->hwPhiAtVtx());
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
      for (int i = RMUON; i <= RIDX; ++i) {
        errorSummaryDen->Fill(i);
      }

      bool muonMismatch = false;
      if (muonIt1->hwPt() != muonIt2->hwPt()) {
        muonMismatch = true;
        summary->Fill(PTBAD);
        errorSummaryNum->Fill(RPT);
      }
      if (muonIt1->hwEta() != muonIt2->hwEta()) {
        muonMismatch = true;
        summary->Fill(ETABAD);
        errorSummaryNum->Fill(RETA);
      }
      if (muonIt1->hwPhi() != muonIt2->hwPhi()) {
        muonMismatch = true;
        summary->Fill(PHIBAD);
        errorSummaryNum->Fill(RPHI);
      }
      if (muonIt1->hwEtaAtVtx() != muonIt2->hwEtaAtVtx()) {
        muonMismatch = true;
        summary->Fill(ETAATVTXBAD);
        errorSummaryNum->Fill(RETAATVTX);
      }
      if (muonIt1->hwPhiAtVtx() != muonIt2->hwPhiAtVtx()) {
        muonMismatch = true;
        summary->Fill(PHIATVTXBAD);
        errorSummaryNum->Fill(RPHIATVTX);
      }
      if (muonIt1->hwCharge() != muonIt2->hwCharge()) {
        muonMismatch = true;
        summary->Fill(CHARGEBAD);
        errorSummaryNum->Fill(RCHARGE);
      }
      if (muonIt1->hwChargeValid() != muonIt2->hwChargeValid()) {
        muonMismatch = true;
        summary->Fill(CHARGEVALBAD);
        errorSummaryNum->Fill(RCHARGEVAL);
      }
      if (muonIt1->hwQual() != muonIt2->hwQual()) {
        muonMismatch = true;
        summary->Fill(QUALBAD);
        errorSummaryNum->Fill(RQUAL);
      }
      if (muonIt1->hwIso() != muonIt2->hwIso()) {
        muonMismatch = true;
        summary->Fill(ISOBAD);
        errorSummaryNum->Fill(RISO);
      }
      if (muonIt1->tfMuonIndex() != muonIt2->tfMuonIndex()) {
        muonMismatch = true;
        summary->Fill(IDXBAD);
        errorSummaryNum->Fill(RIDX);
      }

      if (muonMismatch) {
        errorSummaryNum->Fill(RMUON);

        muColl1hwPt->Fill(muonIt1->hwPt());
        muColl1hwEta->Fill(muonIt1->hwEta());
        muColl1hwPhi->Fill(muonIt1->hwPhi());
        muColl1hwEtaAtVtx->Fill(muonIt1->hwEtaAtVtx());
        muColl1hwPhiAtVtx->Fill(muonIt1->hwPhiAtVtx());
        muColl1hwCharge->Fill(muonIt1->hwCharge());
        muColl1hwChargeValid->Fill(muonIt1->hwChargeValid());
        muColl1hwQual->Fill(muonIt1->hwQual());
        muColl1hwIso->Fill(muonIt1->hwIso());
        muColl1Index->Fill(muonIt1->tfMuonIndex());

        muColl2hwPt->Fill(muonIt2->hwPt());
        muColl2hwEta->Fill(muonIt2->hwEta());
        muColl2hwPhi->Fill(muonIt2->hwPhi());
        muColl2hwEtaAtVtx->Fill(muonIt2->hwEtaAtVtx());
        muColl2hwPhiAtVtx->Fill(muonIt2->hwPhiAtVtx());
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

