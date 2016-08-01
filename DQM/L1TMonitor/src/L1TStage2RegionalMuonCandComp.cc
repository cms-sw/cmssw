#include "DQM/L1TMonitor/interface/L1TStage2RegionalMuonCandComp.h"


L1TStage2RegionalMuonCandComp::L1TStage2RegionalMuonCandComp(const edm::ParameterSet& ps)
    : muonToken1(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("regionalMuonCollection1"))),
      muonToken2(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("regionalMuonCollection2"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      muonColl1Title(ps.getUntrackedParameter<std::string>("regionalMuonCollection1Title", "Regional muon collection 1")),
      muonColl2Title(ps.getUntrackedParameter<std::string>("regionalMuonCollection2Title", "Regional muon collection 2")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false))
{
}

L1TStage2RegionalMuonCandComp::~L1TStage2RegionalMuonCandComp() {}

void L1TStage2RegionalMuonCandComp::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2RegionalMuonCandComp::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2RegionalMuonCandComp::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  // Subsystem Monitoring and Muon Output
  ibooker.setCurrentFolder(monitorDir);

  summary = ibooker.book1D("summary", "Summary", 17, 1, 18); // range to match bin numbering
  summary->setBinLabel(BXRANGEGOOD, "BX range match", 1);
  summary->setBinLabel(BXRANGEBAD, "BX range mismatch", 1);
  summary->setBinLabel(NMUONGOOD, "muon collection size match", 1);
  summary->setBinLabel(NMUONBAD, "muon collection size mismatch", 1);
  summary->setBinLabel(MUONALL, "# muons", 1);
  summary->setBinLabel(MUONGOOD, "# matching muons", 1);
  summary->setBinLabel(PTBAD, "p_{T} mismatch", 1);
  summary->setBinLabel(ETABAD, "#eta mismatch", 1);
  summary->setBinLabel(LOCALPHIBAD, "local #phi mismatch", 1);
  summary->setBinLabel(SIGNBAD, "sign mismatch", 1);
  summary->setBinLabel(SIGNVALBAD, "sign valid mismatch", 1);
  summary->setBinLabel(QUALBAD, "quality mismatch", 1);
  summary->setBinLabel(HFBAD, "HF bit mismatch", 1);
  summary->setBinLabel(LINKBAD, "link mismatch", 1);
  summary->setBinLabel(PROCBAD, "processor mismatch", 1);
  summary->setBinLabel(TFBAD, "track finder type mismatch", 1);
  summary->setBinLabel(TRACKADDRBAD, "track address mismatch", 1);

  muColl1BxRange = ibooker.book1D("muColl1BxRange", (muonColl1Title+" mismatching BX range").c_str(), 5, -2.5, 2.5);
  muColl1BxRange->setAxisTitle("BX range", 1);
  muColl1nMu = ibooker.book1D("muColl1nMu", (muonColl1Title+" mismatching muon multiplicity").c_str(), 37, -0.5, 36.5);
  muColl1nMu->setAxisTitle("Muon multiplicity", 1);
  muColl1hwPt = ibooker.book1D("muColl1hwPt", (muonColl1Title+" mismatching muon p_{T}").c_str(), 512, -0.5, 511.5);
  muColl1hwPt->setAxisTitle("Hardware p_{T}", 1);
  muColl1hwEta = ibooker.book1D("muColl1hwEta", (muonColl1Title+" mismatching muon #eta").c_str(), 512, -256.5, 255.5);
  muColl1hwEta->setAxisTitle("Hardware #eta", 1);
  muColl1hwPhi = ibooker.book1D("muColl1hwPhi", (muonColl1Title+" mismatching muon #phi").c_str(), 256, -128.5, 127.5);
  muColl1hwPhi->setAxisTitle("Hardware #phi", 1);
  muColl1hwSign = ibooker.book1D("muColl1hwSign", (muonColl1Title+" mismatching muon sign").c_str(), 2, -0.5, 1.5);
  muColl1hwSign->setAxisTitle("Hardware sign", 1);
  muColl1hwSignValid = ibooker.book1D("muColl1hwSignValid", (muonColl1Title+" mismatching muon sign valid").c_str(), 2, -0.5, 1.5);
  muColl1hwSignValid->setAxisTitle("Hardware sign valid", 1);
  muColl1hwQual = ibooker.book1D("muColl1hwQual", (muonColl1Title+" mismatching muon quality").c_str(), 16, -0.5, 15.5);
  muColl1hwQual->setAxisTitle("Hardware quality", 1);
  muColl1link = ibooker.book1D("muColl1link", (muonColl1Title+" mismatching link").c_str(), 36, 35.5, 71.5);
  muColl1link->setAxisTitle("Link", 1);
  muColl1processor = ibooker.book1D("muColl1processor", (muonColl1Title+" mismatching processor").c_str(), 12, -0.5, 15.5);
  muColl1processor->setAxisTitle("Processor", 1);
  muColl1trackFinderType = ibooker.book1D("muColl1trackFinderType", (muonColl1Title+" mismatching track finder type").c_str(), 3, -0.5, 2.5);
  muColl1trackFinderType->setAxisTitle("Track finder type", 1);
  muColl1trackFinderType->setBinLabel(BMTFBIN, "BMTF", 1);
  muColl1trackFinderType->setBinLabel(OMTFBIN, "OMTF", 1);
  muColl1trackFinderType->setBinLabel(EMTFBIN, "EMTF", 1);
  muColl1hwHF = ibooker.book1D("muColl1hwHF", (muonColl1Title+" mismatching muon halo/fine-eta bit").c_str(), 2, -0.5, 1.5);
  muColl1hwHF->setAxisTitle("Hardware H/F bit", 1);

  muColl2BxRange = ibooker.book1D("muColl2BxRange", (muonColl2Title+" mismatching BX range").c_str(), 5, -2.5, 2.5);
  muColl2BxRange->setAxisTitle("BX range", 1);
  muColl2nMu = ibooker.book1D("muColl2nMu", (muonColl2Title+" mismatching muon multiplicity").c_str(), 37, -0.5, 36.5);
  muColl2nMu->setAxisTitle("Muon multiplicity", 1);
  muColl2hwPt = ibooker.book1D("muColl2hwPt", (muonColl2Title+" mismatching muon p_{T}").c_str(), 512, -0.5, 511.5);
  muColl2hwPt->setAxisTitle("Hardware p_{T}", 1);
  muColl2hwEta = ibooker.book1D("muColl2hwEta", (muonColl2Title+" mismatching muon #eta").c_str(), 512, -256.5, 255.5);
  muColl2hwEta->setAxisTitle("Hardware #eta", 1);
  muColl2hwPhi = ibooker.book1D("muColl2hwPhi", (muonColl2Title+" mismatching muon #phi").c_str(), 256, -128.5, 127.5);
  muColl2hwPhi->setAxisTitle("Hardware #phi", 1);
  muColl2hwSign = ibooker.book1D("muColl2hwSign", (muonColl2Title+" mismatching muon sign").c_str(), 2, -0.5, 1.5);
  muColl2hwSign->setAxisTitle("Hardware sign", 1);
  muColl2hwSignValid = ibooker.book1D("muColl2hwSignValid", (muonColl2Title+" mismatching muon sign valid").c_str(), 2, -0.5, 1.5);
  muColl2hwSignValid->setAxisTitle("Hardware sign valid", 1);
  muColl2hwQual = ibooker.book1D("muColl2hwQual", (muonColl2Title+" mismatching muon quality").c_str(), 16, -0.5, 15.5);
  muColl2hwQual->setAxisTitle("Hardware quality", 1);
  muColl2link = ibooker.book1D("muColl2link", (muonColl2Title+" mismatching link").c_str(), 36, 35.5, 71.5);
  muColl2link->setAxisTitle("Link", 1);
  muColl2processor = ibooker.book1D("muColl2processor", (muonColl2Title+" mismatching processor").c_str(), 12, -0.5, 15.5);
  muColl2processor->setAxisTitle("Processor", 1);
  muColl2trackFinderType = ibooker.book1D("muColl2trackFinderType", (muonColl2Title+" mismatching track finder type").c_str(), 3, -0.5, 2.5);
  muColl2trackFinderType->setAxisTitle("Track finder type", 1);
  muColl2trackFinderType->setBinLabel(BMTFBIN, "BMTF", 1);
  muColl2trackFinderType->setBinLabel(OMTFBIN, "OMTF", 1);
  muColl2trackFinderType->setBinLabel(EMTFBIN, "EMTF", 1);
  muColl2hwHF = ibooker.book1D("muColl2hwHF", (muonColl2Title+" mismatching muon halo/fine-eta bit").c_str(), 2, -0.5, 1.5);
  muColl2hwHF->setAxisTitle("Hardware H/F bit", 1);

}

void L1TStage2RegionalMuonCandComp::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2RegionalMuonCandComp") << "L1TStage2RegionalMuonCandComp: analyze..." << std::endl;

  edm::Handle<l1t::RegionalMuonCandBxCollection> muonBxColl1;
  edm::Handle<l1t::RegionalMuonCandBxCollection> muonBxColl2;
  e.getByToken(muonToken1, muonBxColl1);
  e.getByToken(muonToken2, muonBxColl2);

  int bxRange1 = muonBxColl1->getLastBX() - muonBxColl1->getFirstBX();
  int bxRange2 = muonBxColl2->getLastBX() - muonBxColl2->getFirstBX();
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

    l1t::RegionalMuonCandBxCollection::const_iterator muonIt1;
    l1t::RegionalMuonCandBxCollection::const_iterator muonIt2;

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
          muColl1hwSign->Fill(muonIt1->hwSign());
          muColl1hwSignValid->Fill(muonIt1->hwSignValid());
          muColl1hwQual->Fill(muonIt1->hwQual());
          muColl1link->Fill(muonIt1->link());
          muColl1processor->Fill(muonIt1->processor());
          muColl1trackFinderType->Fill(muonIt1->trackFinderType());
          muColl1hwHF->Fill(muonIt1->hwHF());
        }
      } else {
        muonIt2 = muonBxColl2->begin(iBx) + muonBxColl1->size(iBx);
        for (; muonIt2 != muonBxColl2->end(iBx); ++muonIt2) {
          muColl2hwPt->Fill(muonIt2->hwPt());
          muColl2hwEta->Fill(muonIt2->hwEta());
          muColl2hwPhi->Fill(muonIt2->hwPhi());
          muColl2hwSign->Fill(muonIt2->hwSign());
          muColl2hwSignValid->Fill(muonIt2->hwSignValid());
          muColl2hwQual->Fill(muonIt2->hwQual());
          muColl2link->Fill(muonIt1->link());
          muColl2processor->Fill(muonIt1->processor());
          muColl2trackFinderType->Fill(muonIt1->trackFinderType());
          muColl2hwHF->Fill(muonIt1->hwHF());
        }
      }
    } else {
      summary->Fill(NMUONGOOD);
    }

    muonIt1 = muonBxColl1->begin(iBx);
    muonIt2 = muonBxColl2->begin(iBx);
    while(muonIt1 != muonBxColl1->end(iBx) && muonIt2 != muonBxColl1->end(iBx)) {
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
        summary->Fill(LOCALPHIBAD);
      }
      if (muonIt1->hwSign() != muonIt2->hwSign()) {
        muonMismatch = true;
        summary->Fill(SIGNBAD);
      }
      if (muonIt1->hwSignValid() != muonIt2->hwSignValid()) {
        muonMismatch = true;
        summary->Fill(SIGNVALBAD);
      }
      if (muonIt1->hwQual() != muonIt2->hwQual()) {
        muonMismatch = true;
        summary->Fill(QUALBAD);
      }
      if (muonIt1->link() != muonIt2->link()) {
        muonMismatch = true;
        summary->Fill(LINKBAD);
      }
      if (muonIt1->processor() != muonIt2->processor()) {
        muonMismatch = true;
        summary->Fill(PROCBAD);
      }
      if (muonIt1->trackFinderType() != muonIt2->trackFinderType()) {
        muonMismatch = true;
        summary->Fill(TFBAD);
      }
      // check track address
      const std::map<int, int> muon1TrackAddr = muonIt1->trackAddress();
      std::map<int, int> muon2TrackAddr = muonIt2->trackAddress();
      bool badTrackAddr = false;
      if (muon1TrackAddr.size() == muon2TrackAddr.size()) {
        for (std::map<int, int>::const_iterator trIt1 = muon1TrackAddr.begin(); trIt1 != muon1TrackAddr.end(); ++trIt1) {
          if (muon2TrackAddr.find(trIt1->first) == muon2TrackAddr.end()) {
            badTrackAddr = true;
            break;
          } else if (muon2TrackAddr[trIt1->first] != trIt1->second) {
            badTrackAddr = true;
            break;
          }
        }
      } else {
        badTrackAddr = true;
      }
      if (badTrackAddr) {
        muonMismatch = true;
        summary->Fill(TRACKADDRBAD);
      }

      if (muonMismatch) {
        muColl1hwPt->Fill(muonIt1->hwPt());
        muColl1hwEta->Fill(muonIt1->hwEta());
        muColl1hwPhi->Fill(muonIt1->hwPhi());
        muColl1hwSign->Fill(muonIt1->hwSign());
        muColl1hwSignValid->Fill(muonIt1->hwSignValid());
        muColl1hwQual->Fill(muonIt1->hwQual());
        muColl1link->Fill(muonIt1->link());
        muColl1processor->Fill(muonIt1->processor());
        muColl1trackFinderType->Fill(muonIt1->trackFinderType());
        muColl1hwHF->Fill(muonIt1->hwHF());

        muColl2hwPt->Fill(muonIt2->hwPt());
        muColl2hwEta->Fill(muonIt2->hwEta());
        muColl2hwPhi->Fill(muonIt2->hwPhi());
        muColl2hwSign->Fill(muonIt2->hwSign());
        muColl2hwSignValid->Fill(muonIt2->hwSignValid());
        muColl2hwQual->Fill(muonIt2->hwQual());
        muColl2link->Fill(muonIt1->link());
        muColl2processor->Fill(muonIt1->processor());
        muColl2trackFinderType->Fill(muonIt1->trackFinderType());
        muColl2hwHF->Fill(muonIt1->hwHF());
      } else {
        summary->Fill(MUONGOOD);
      }

      ++muonIt1;
      ++muonIt2;
    }
  }
}

