#include "DQM/L1TMonitor/interface/L1TStage2RegionalMuonCandComp.h"


L1TStage2RegionalMuonCandComp::L1TStage2RegionalMuonCandComp(const edm::ParameterSet& ps)
    : muonToken1(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("regionalMuonCollection1"))),
      muonToken2(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("regionalMuonCollection2"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir")),
      muonColl1Title(ps.getUntrackedParameter<std::string>("regionalMuonCollection1Title")),
      muonColl2Title(ps.getUntrackedParameter<std::string>("regionalMuonCollection2Title")),
      summaryTitle(ps.getUntrackedParameter<std::string>("summaryTitle")),
      ignoreBadTrkAddr(ps.getUntrackedParameter<bool>("ignoreBadTrackAddress")),
      ignoreBin(ps.getUntrackedParameter<std::vector<int>>("ignoreBin")),
      verbose(ps.getUntrackedParameter<bool>("verbose")),
      isBmtf(ps.getUntrackedParameter<bool>("isBmtf"))
{
  // First include all bins
  for (unsigned int i = 1; i <= RPT2; i++) { 
    incBin[i] = true;
  }
  // Then check the list of bins to ignore
  for (const auto& i : ignoreBin) {
    if (i > 0 && i <= RPT2) {
      incBin[i] = false;
    }
  }

}

L1TStage2RegionalMuonCandComp::~L1TStage2RegionalMuonCandComp() {}

void L1TStage2RegionalMuonCandComp::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("regionalMuonCollection1")->setComment("L1T RegionalMuonCand collection 1");
  desc.add<edm::InputTag>("regionalMuonCollection2")->setComment("L1T RegionalMuonCand collection 2");
  desc.addUntracked<std::string>("monitorDir", "")->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<std::string>("regionalMuonCollection1Title", "Regional muon collection 1")->setComment("Histogram title for first collection.");
  desc.addUntracked<std::string>("regionalMuonCollection2Title", "Regional muon collection 2")->setComment("Histogram title for second collection.");
  desc.addUntracked<std::string>("summaryTitle", "Summary")->setComment("Title of summary histogram.");
  desc.addUntracked<bool>("ignoreBadTrackAddress", false)->setComment("Ignore muon track address mismatches.");
  desc.addUntracked<std::vector<int>>("ignoreBin", std::vector<int>())->setComment("List of bins to ignore");
  desc.addUntracked<bool>("verbose", false);
  desc.addUntracked<bool>("isBmtf", false);
  descriptions.add("l1tStage2RegionalMuonCandComp", desc);
}

void L1TStage2RegionalMuonCandComp::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}


void L1TStage2RegionalMuonCandComp::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  std::string trkAddrIgnoreText = "";
  if (ignoreBadTrkAddr) {
    trkAddrIgnoreText = " (Bad track addresses ignored)";
  }

  int nbins = 17;
  if (isBmtf) {
    nbins += 2;
  }


  // Subsystem Monitoring and Muon Output
  ibooker.setCurrentFolder(monitorDir);

  summary = ibooker.book1D("summary", (summaryTitle+trkAddrIgnoreText).c_str(), nbins, 1, nbins+1); // range to match bin numbering
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
  if (isBmtf){
    summary->setBinLabel(DXYBAD, "DXY mismatch", 1);
    summary->setBinLabel(PT2BAD, "P_{T}2 mismatch", 1);
  }

  int nbinsNum = 14;
  if (isBmtf) {
    nbinsNum += 2;
  }

  errorSummaryNum = ibooker.book1D("errorSummaryNum", (summaryTitle+trkAddrIgnoreText).c_str(), nbinsNum, 1, nbinsNum+1); // range to match bin numbering
  errorSummaryNum->setBinLabel(RBXRANGE, "BX range mismatch", 1);
  errorSummaryNum->setBinLabel(RNMUON, "muon collection size mismatch", 1);
  errorSummaryNum->setBinLabel(RMUON, "mismatching muons", 1);
  errorSummaryNum->setBinLabel(RPT, "p_{T} mismatch", 1);
  errorSummaryNum->setBinLabel(RETA, "#eta mismatch", 1);
  errorSummaryNum->setBinLabel(RLOCALPHI, "local #phi mismatch", 1);
  errorSummaryNum->setBinLabel(RSIGN, "sign mismatch", 1);
  errorSummaryNum->setBinLabel(RSIGNVAL, "sign valid mismatch", 1);
  errorSummaryNum->setBinLabel(RQUAL, "quality mismatch", 1);
  errorSummaryNum->setBinLabel(RHF, "HF bit mismatch", 1);
  errorSummaryNum->setBinLabel(RLINK, "link mismatch", 1);
  errorSummaryNum->setBinLabel(RPROC, "processor mismatch", 1);
  errorSummaryNum->setBinLabel(RTF, "track finder type mismatch", 1);
  errorSummaryNum->setBinLabel(RTRACKADDR, "track address mismatch", 1);
  if (isBmtf) {
    errorSummaryNum->setBinLabel(RDXY, "DXY mismatch", 1);
    errorSummaryNum->setBinLabel(RPT2, "P_{T}2 mismatch", 1);
  }

  // Change the label for those bins that will be ignored
  for (int i = 1; i <= errorSummaryNum->getNbinsX(); i++) {
    if (incBin[i]==false) {
      errorSummaryNum->setBinLabel(i, "Ignored", 1);
    }
  }
  // Setting canExtend to false is needed to get the correct behaviour when running multithreaded.
  // Otherwise, when merging the histgrams of the threads, TH1::Merge sums bins that have the same label in one bin.
  // This needs to come after the calls to setBinLabel.
  errorSummaryNum->getTH1F()->GetXaxis()->SetCanExtend(false);

  errorSummaryDen = ibooker.book1D("errorSummaryDen", "denominators", nbinsNum, 1, nbinsNum+1); // range to match bin numbering
  errorSummaryDen->setBinLabel(RBXRANGE, "# events", 1);
  errorSummaryDen->setBinLabel(RNMUON, "# muon collections", 1);
  for (int i = RMUON; i <= errorSummaryDen->getNbinsX(); ++i) {
    errorSummaryDen->setBinLabel(i, "# muons", 1);
  }
  // Needed for correct histogram summing in multithreaded running.
  errorSummaryDen->getTH1F()->GetXaxis()->SetCanExtend(false);

  muColl1BxRange = ibooker.book1D("muBxRangeColl1", (muonColl1Title+" mismatching BX range").c_str(), 11, -5.5, 5.5);
  muColl1BxRange->setAxisTitle("BX range", 1);
  muColl1nMu = ibooker.book1D("nMuColl1", (muonColl1Title+" mismatching muon multiplicity").c_str(), 37, -0.5, 36.5);
  muColl1nMu->setAxisTitle("Muon multiplicity", 1);
  muColl1hwPt = ibooker.book1D("muHwPtColl1", (muonColl1Title+" mismatching muon p_{T}"+trkAddrIgnoreText).c_str(), 512, -0.5, 511.5);
  muColl1hwPt->setAxisTitle("Hardware p_{T}", 1);
  muColl1hwEta = ibooker.book1D("muHwEtaColl1", (muonColl1Title+" mismatching muon #eta"+trkAddrIgnoreText).c_str(), 512, -256.5, 255.5);
  muColl1hwEta->setAxisTitle("Hardware #eta", 1);
  muColl1hwPhi = ibooker.book1D("muHwPhiColl1", (muonColl1Title+" mismatching muon #phi"+trkAddrIgnoreText).c_str(), 256, -128.5, 127.5);
  muColl1hwPhi->setAxisTitle("Hardware #phi", 1);
  muColl1hwSign = ibooker.book1D("muHwSignColl1", (muonColl1Title+" mismatching muon sign"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  muColl1hwSign->setAxisTitle("Hardware sign", 1);
  muColl1hwSignValid = ibooker.book1D("muHwSignValidColl1", (muonColl1Title+" mismatching muon sign valid"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  muColl1hwSignValid->setAxisTitle("Hardware sign valid", 1);
  muColl1hwQual = ibooker.book1D("muHwQualColl1", (muonColl1Title+" mismatching muon quality"+trkAddrIgnoreText).c_str(), 16, -0.5, 15.5);
  muColl1hwQual->setAxisTitle("Hardware quality", 1);
  muColl1link = ibooker.book1D("muLinkColl1", (muonColl1Title+" mismatching muon link"+trkAddrIgnoreText).c_str(), 36, 35.5, 71.5);
  muColl1link->setAxisTitle("Link", 1);
  muColl1processor = ibooker.book1D("muProcessorColl1", (muonColl1Title+" mismatching muon processor"+trkAddrIgnoreText).c_str(), 12, -0.5, 11.5);
  muColl1processor->setAxisTitle("Processor", 1);
  muColl1trackFinderType = ibooker.book1D("muTrackFinderTypeColl1", (muonColl1Title+" mismatching muon track finder type"+trkAddrIgnoreText).c_str(), 5, -0.5, 4.5);
  muColl1trackFinderType->setAxisTitle("Track finder type", 1);
  muColl1trackFinderType->setBinLabel(BMTFBIN, "BMTF", 1);
  muColl1trackFinderType->setBinLabel(OMTFNEGBIN, "OMTF-", 1);
  muColl1trackFinderType->setBinLabel(OMTFPOSBIN, "OMTF+", 1);
  muColl1trackFinderType->setBinLabel(EMTFNEGBIN, "EMTF-", 1);
  muColl1trackFinderType->setBinLabel(EMTFPOSBIN, "EMTF+", 1);
  muColl1hwHF = ibooker.book1D("muHwHFColl1", (muonColl1Title+" mismatching muon halo/fine-eta bit"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  muColl1hwHF->setAxisTitle("Hardware H/F bit", 1);
  muColl1TrkAddrSize = ibooker.book1D("muTrkAddrSizeColl1", (muonColl1Title+" mismatching muon number of track address keys"+trkAddrIgnoreText).c_str(), 11, -0.5, 10.5);
  muColl1TrkAddrSize->setAxisTitle("number of keys", 1);
  muColl1TrkAddr = ibooker.book2D("muTrkAddrColl1", (muonColl1Title+" mismatching muon track address"+trkAddrIgnoreText).c_str(), 10, -0.5, 9.5, 16, -0.5, 15.5);
  muColl1TrkAddr->setAxisTitle("key", 1);
  muColl1TrkAddr->setAxisTitle("value", 2);
  if (isBmtf) {
    muColl1hwDXY = ibooker.book1D("muhwDXYColl1", (muonColl1Title+" HW DXY"+trkAddrIgnoreText).c_str(), 4, 0, 4);
    muColl1hwDXY->setAxisTitle("Hardware DXY",1);
    muColl1hwPt2 = ibooker.book1D("muhwPt2Coll1", (muonColl1Title+"HW p_{T}2"+trkAddrIgnoreText).c_str(), 512, -0.5, 511.5);
    muColl1hwPt2->setAxisTitle("Hardware P_{T}2",1);
  }

  muColl2BxRange = ibooker.book1D("muBxRangeColl2", (muonColl2Title+" mismatching BX range").c_str(), 11, -5.5, 5.5);
  muColl2BxRange->setAxisTitle("BX range", 1);
  muColl2nMu = ibooker.book1D("nMuColl2", (muonColl2Title+" mismatching muon multiplicity").c_str(), 37, -0.5, 36.5);
  muColl2nMu->setAxisTitle("Muon multiplicity", 1);
  muColl2hwPt = ibooker.book1D("muHwPtColl2", (muonColl2Title+" mismatching muon p_{T}"+trkAddrIgnoreText).c_str(), 512, -0.5, 511.5);
  muColl2hwPt->setAxisTitle("Hardware p_{T}", 1);
  muColl2hwEta = ibooker.book1D("muHwEtaColl2", (muonColl2Title+" mismatching muon #eta"+trkAddrIgnoreText).c_str(), 512, -256.5, 255.5);
  muColl2hwEta->setAxisTitle("Hardware #eta", 1);
  muColl2hwPhi = ibooker.book1D("muHwPhiColl2", (muonColl2Title+" mismatching muon #phi"+trkAddrIgnoreText).c_str(), 256, -128.5, 127.5);
  muColl2hwPhi->setAxisTitle("Hardware #phi", 1);
  muColl2hwSign = ibooker.book1D("muHwSignColl2", (muonColl2Title+" mismatching muon sign"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  muColl2hwSign->setAxisTitle("Hardware sign", 1);
  muColl2hwSignValid = ibooker.book1D("muHwSignValidColl2", (muonColl2Title+" mismatching muon sign valid"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  muColl2hwSignValid->setAxisTitle("Hardware sign valid", 1);
  muColl2hwQual = ibooker.book1D("muHwQualColl2", (muonColl2Title+" mismatching muon quality"+trkAddrIgnoreText).c_str(), 16, -0.5, 15.5);
  muColl2hwQual->setAxisTitle("Hardware quality", 1);
  muColl2link = ibooker.book1D("muLinkColl2", (muonColl2Title+" mismatching muon link"+trkAddrIgnoreText).c_str(), 36, 35.5, 71.5);
  muColl2link->setAxisTitle("Link", 1);
  muColl2processor = ibooker.book1D("muProcessorColl2", (muonColl2Title+" mismatching muon processor"+trkAddrIgnoreText).c_str(), 12, -0.5, 11.5);
  muColl2processor->setAxisTitle("Processor", 1);
  muColl2trackFinderType = ibooker.book1D("muTrackFinderTypeColl2", (muonColl2Title+" mismatching muon track finder type"+trkAddrIgnoreText).c_str(), 5, -0.5, 4.5);
  muColl2trackFinderType->setAxisTitle("Track finder type", 1);
  muColl2trackFinderType->setBinLabel(BMTFBIN, "BMTF", 1);
  muColl2trackFinderType->setBinLabel(OMTFNEGBIN, "OMTF-", 1);
  muColl2trackFinderType->setBinLabel(OMTFPOSBIN, "OMTF+", 1);
  muColl2trackFinderType->setBinLabel(EMTFNEGBIN, "EMTF-", 1);
  muColl2trackFinderType->setBinLabel(EMTFPOSBIN, "EMTF+", 1);
  muColl2hwHF = ibooker.book1D("muHwHFColl2", (muonColl2Title+" mismatching muon halo/fine-eta bit"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  muColl2hwHF->setAxisTitle("Hardware H/F bit", 1);
  muColl2TrkAddrSize = ibooker.book1D("muTrkAddrSizeColl2", (muonColl2Title+" mismatching muon number of track address keys"+trkAddrIgnoreText).c_str(), 11, -0.5, 10.5);
  muColl2TrkAddrSize->setAxisTitle("number of keys", 1);
  muColl2TrkAddr = ibooker.book2D("muTrkAddrColl2", (muonColl2Title+" mismatching muon track address"+trkAddrIgnoreText).c_str(), 10, -0.5, 9.5, 16, -0.5, 15.5);
  muColl2TrkAddr->setAxisTitle("key", 1);
  muColl2TrkAddr->setAxisTitle("value", 2);
  if (isBmtf) {
    muColl2hwDXY = ibooker.book1D("muhwDXYColl2", (muonColl2Title+" HW DXY"+trkAddrIgnoreText).c_str(), 4, 0, 4);
    muColl2hwDXY->setAxisTitle("Hardware DXY",1);
    muColl2hwPt2 = ibooker.book1D("muhwPt2Coll2", (muonColl2Title+"HW p_{T}2"+trkAddrIgnoreText).c_str(), 512, -0.5, 511.5);
    muColl2hwPt2->setAxisTitle("Hardware P_{T}2",1);
  }
}

void L1TStage2RegionalMuonCandComp::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2RegionalMuonCandComp") << "L1TStage2RegionalMuonCandComp: analyze..." << std::endl;

  edm::Handle<l1t::RegionalMuonCandBxCollection> muonBxColl1;
  edm::Handle<l1t::RegionalMuonCandBxCollection> muonBxColl2;
  e.getByToken(muonToken1, muonBxColl1);
  e.getByToken(muonToken2, muonBxColl2);

  errorSummaryDen->Fill(RBXRANGE);
  int bxRange1 = muonBxColl1->getLastBX() - muonBxColl1->getFirstBX() + 1;
  int bxRange2 = muonBxColl2->getLastBX() - muonBxColl2->getFirstBX() + 1;
  if (bxRange1 != bxRange2) {
    summary->Fill(BXRANGEBAD);
    if (incBin[RBXRANGE]) errorSummaryNum->Fill(RBXRANGE);
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

    errorSummaryDen->Fill(RNMUON);
    // check number of muons
    if (muonBxColl1->size(iBx) != muonBxColl2->size(iBx)) {
      summary->Fill(NMUONBAD);
      if (incBin[RNMUON]) errorSummaryNum->Fill(RNMUON);
      muColl1nMu->Fill(muonBxColl1->size(iBx));
      muColl2nMu->Fill(muonBxColl2->size(iBx));

      if (muonBxColl1->size(iBx) > muonBxColl2->size(iBx)) {
        muonIt1 = muonBxColl1->begin(iBx) + muonBxColl2->size(iBx);
        const std::map<int, int> muon1TrackAddr = muonIt1->trackAddress();
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
          muColl1TrkAddrSize->Fill(muon1TrackAddr.size());
          if (isBmtf){
            muColl1hwDXY->Fill(muonIt1->hwDXY());
            muColl1hwPt2->Fill(muonIt1->hwPt2());         
	  }
          for (std::map<int, int>::const_iterator trIt1 = muon1TrackAddr.begin(); trIt1 != muon1TrackAddr.end(); ++trIt1) {
            muColl1TrkAddr->Fill(trIt1->first, trIt1->second);
          }
        }
      } else {
        muonIt2 = muonBxColl2->begin(iBx) + muonBxColl1->size(iBx);
        const std::map<int, int> muon2TrackAddr = muonIt2->trackAddress();
        for (; muonIt2 != muonBxColl2->end(iBx); ++muonIt2) {
          muColl2hwPt->Fill(muonIt2->hwPt());
          muColl2hwEta->Fill(muonIt2->hwEta());
          muColl2hwPhi->Fill(muonIt2->hwPhi());
          muColl2hwSign->Fill(muonIt2->hwSign());
          muColl2hwSignValid->Fill(muonIt2->hwSignValid());
          muColl2hwQual->Fill(muonIt2->hwQual());
          muColl2link->Fill(muonIt2->link());
          muColl2processor->Fill(muonIt2->processor());
          muColl2trackFinderType->Fill(muonIt2->trackFinderType());
          muColl2hwHF->Fill(muonIt2->hwHF());
          muColl2TrkAddrSize->Fill(muon2TrackAddr.size());
          if (isBmtf){
            muColl2hwDXY->Fill(muonIt2->hwDXY());
            muColl2hwPt2->Fill(muonIt2->hwPt2());
	  }
          for (std::map<int, int>::const_iterator trIt2 = muon2TrackAddr.begin(); trIt2 != muon2TrackAddr.end(); ++trIt2) {
            muColl2TrkAddr->Fill(trIt2->first, trIt2->second);
          }
        }
      }
    } else {
      summary->Fill(NMUONGOOD);
    }

    muonIt1 = muonBxColl1->begin(iBx);
    muonIt2 = muonBxColl2->begin(iBx);
    //std::cout << "Analysing muons from BX " << iBx << std::endl;
    while(muonIt1 != muonBxColl1->end(iBx) && muonIt2 != muonBxColl2->end(iBx)) {
      //std::cout << "Coll 1 muon: hwPt=" << muonIt1->hwPt() << ", hwEta=" << muonIt1->hwEta() << ", hwPhi=" << muonIt1->hwPhi()
      //          << ", hwSign=" << muonIt1->hwSign() << ", hwSignValid=" << muonIt1->hwSignValid()
      //          << ", hwQual=" << muonIt1->hwQual() << ", link=" << muonIt1->link() << ", processor=" << muonIt1->processor()
      //          << ", trackFinderType=" << muonIt1->trackFinderType() << std::endl;
      //std::cout << "Coll 2 muon: hwPt=" << muonIt2->hwPt() << ", hwEta=" << muonIt2->hwEta() << ", hwPhi=" << muonIt2->hwPhi()
      //          << ", hwSign=" << muonIt2->hwSign() << ", hwSignValid=" << muonIt2->hwSignValid()
      //          << ", hwQual=" << muonIt2->hwQual() << ", link=" << muonIt2->link() << ", processor=" << muonIt2->processor()
      //          << ", trackFinderType=" << muonIt2->trackFinderType() << std::endl;
      summary->Fill(MUONALL);
      for (int i = RMUON; i <= errorSummaryDen->getNbinsX(); ++i) {
        errorSummaryDen->Fill(i);
      }

      bool muonMismatch = false;    // All muon mismatches
      bool muonSelMismatch = false; // Muon mismatches excluding ignored bins
      if (muonIt1->hwPt() != muonIt2->hwPt()) {
        muonMismatch = true;
        summary->Fill(PTBAD);
        if (incBin[RPT]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RPT);
        }
      }
      if (muonIt1->hwEta() != muonIt2->hwEta()) {
        muonMismatch = true;
        summary->Fill(ETABAD);
        if (incBin[RETA]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RETA);
        }
      }
      if (muonIt1->hwPhi() != muonIt2->hwPhi()) {
        muonMismatch = true;
        summary->Fill(LOCALPHIBAD);
        if (incBin[RLOCALPHI]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RLOCALPHI);
        }
      }
      if (muonIt1->hwSign() != muonIt2->hwSign()) {
        muonMismatch = true;
        summary->Fill(SIGNBAD);
        if (incBin[RSIGN]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RSIGN);
        }
      }
      if (muonIt1->hwSignValid() != muonIt2->hwSignValid()) {
        muonMismatch = true;
        summary->Fill(SIGNVALBAD);
        if (incBin[RSIGNVAL]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RSIGNVAL);
        }
      }
      if (muonIt1->hwQual() != muonIt2->hwQual()) {
        muonMismatch = true;
        summary->Fill(QUALBAD);
        if (incBin[RQUAL]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RQUAL);
        }
      }
      if (muonIt1->hwHF() != muonIt2->hwHF()) {
        muonMismatch = true;
        summary->Fill(HFBAD);
        if (incBin[RHF]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RHF);
        }
      }
      if (muonIt1->link() != muonIt2->link()) {
        muonMismatch = true;
        summary->Fill(LINKBAD);
        if (incBin[RLINK]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RLINK);
        }
      }
      if (muonIt1->processor() != muonIt2->processor()) {
        muonMismatch = true;
        summary->Fill(PROCBAD);
        if (incBin[RPROC]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RPROC);
        }
      }
      if (muonIt1->trackFinderType() != muonIt2->trackFinderType()) {
        muonMismatch = true;
        summary->Fill(TFBAD);
        if (incBin[RTF]) {
          muonSelMismatch = true;
          errorSummaryNum->Fill(RTF);
        }
      }
      // check track address
      const std::map<int, int> muon1TrackAddr = muonIt1->trackAddress();
      std::map<int, int> muon2TrackAddr = muonIt2->trackAddress();
      bool badTrackAddr = false;
      if (muon1TrackAddr.size() == muon2TrackAddr.size()) {
        for (std::map<int, int>::const_iterator trIt1 = muon1TrackAddr.begin(); trIt1 != muon1TrackAddr.end(); ++trIt1) {
          if (muon2TrackAddr.find(trIt1->first) == muon2TrackAddr.end()) { // key does not exist
            badTrackAddr = true;
            break;
          } else if (muon2TrackAddr[trIt1->first] != trIt1->second) { // wrong value for key
            badTrackAddr = true;
            break;
          }
        }
      } else {
        badTrackAddr = true;
      }
      if (badTrackAddr) {
        if (!ignoreBadTrkAddr) {
          muonMismatch = true;
          if (incBin[RTRACKADDR]) muonSelMismatch = true;
        }
        summary->Fill(TRACKADDRBAD);
        if (incBin[RTRACKADDR]) errorSummaryNum->Fill(RTRACKADDR);
      }

      if (incBin[RMUON] && muonSelMismatch) {
        errorSummaryNum->Fill(RMUON);
      }

      if(isBmtf) {
        if (muonIt1->hwDXY() != muonIt2->hwDXY()) {
          muonMismatch = true;
          summary->Fill(DXYBAD);
          if (incBin[RDXY]) {
            muonSelMismatch = true;
            errorSummaryNum->Fill(RDXY);
          }
        }
        if (muonIt1->hwPt2() != muonIt2->hwPt2()) {
          muonMismatch = true;
          summary->Fill(PT2BAD);
          if (incBin[RPT2]) {
            muonSelMismatch = true;
            errorSummaryNum->Fill(RPT2);
          }
        }
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
        muColl1TrkAddrSize->Fill(muon1TrackAddr.size());
        if (isBmtf){
          muColl1hwDXY->Fill(muonIt1->hwDXY());
          muColl1hwPt2->Fill(muonIt1->hwPt2());
	}
        for (std::map<int, int>::const_iterator trIt1 = muon1TrackAddr.begin(); trIt1 != muon1TrackAddr.end(); ++trIt1) {
          muColl1TrkAddr->Fill(trIt1->first, trIt1->second);
        }

        muColl2hwPt->Fill(muonIt2->hwPt());
        muColl2hwEta->Fill(muonIt2->hwEta());
        muColl2hwPhi->Fill(muonIt2->hwPhi());
        muColl2hwSign->Fill(muonIt2->hwSign());
        muColl2hwSignValid->Fill(muonIt2->hwSignValid());
        muColl2hwQual->Fill(muonIt2->hwQual());
        muColl2link->Fill(muonIt2->link());
        muColl2processor->Fill(muonIt2->processor());
        muColl2trackFinderType->Fill(muonIt2->trackFinderType());
        muColl2hwHF->Fill(muonIt2->hwHF());
        muColl2TrkAddrSize->Fill(muon2TrackAddr.size());
        if (isBmtf){
          muColl2hwDXY->Fill(muonIt2->hwDXY());
          muColl2hwPt2->Fill(muonIt2->hwPt2());
	}
        for (std::map<int, int>::const_iterator trIt2 = muon2TrackAddr.begin(); trIt2 != muon2TrackAddr.end(); ++trIt2) {
          muColl2TrkAddr->Fill(trIt2->first, trIt2->second);
        }
      } else {
        summary->Fill(MUONGOOD);
      }

      ++muonIt1;
      ++muonIt2;
    }
  }
}

