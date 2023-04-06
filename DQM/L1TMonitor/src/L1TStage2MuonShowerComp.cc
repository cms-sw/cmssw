#include "DQM/L1TMonitor/interface/L1TStage2MuonShowerComp.h"

L1TStage2MuonShowerComp::L1TStage2MuonShowerComp(const edm::ParameterSet& ps)
    : showerToken1_(consumes<l1t::MuonShowerBxCollection>(ps.getParameter<edm::InputTag>("muonShowerCollection1"))),
      showerToken2_(consumes<l1t::MuonShowerBxCollection>(ps.getParameter<edm::InputTag>("muonShowerCollection2"))),
      monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir")),
      showerColl1Title_(ps.getUntrackedParameter<std::string>("muonShowerCollection1Title")),
      showerColl2Title_(ps.getUntrackedParameter<std::string>("muonShowerCollection2Title")),
      summaryTitle_(ps.getUntrackedParameter<std::string>("summaryTitle")),
      ignoreBin_(ps.getUntrackedParameter<std::vector<int>>("ignoreBin")),
      verbose_(ps.getUntrackedParameter<bool>("verbose")) {
  // First include all bins
  for (int i = 1; i <= numErrBins_; i++) {
    incBin_[i] = true;
  }
  // Then check the list of bins to ignore
  for (const auto& i : ignoreBin_) {
    if (i > 0 && i <= numErrBins_) {
      incBin_[i] = false;
    }
  }
}

L1TStage2MuonShowerComp::~L1TStage2MuonShowerComp() {}

void L1TStage2MuonShowerComp::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonShowerCollection1")->setComment("L1T Muon Shower collection 1");
  desc.add<edm::InputTag>("muonShowerCollection2")->setComment("L1T Muon Shower collection 2");
  desc.addUntracked<std::string>("monitorDir", "")
      ->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<std::string>("muonShowerCollection1Title", "Muon Shower collection 1")
      ->setComment("Histogram title for first collection.");
  desc.addUntracked<std::string>("muonShowerCollection2Title", "Muon Shower collection 2")
      ->setComment("Histogram title for second collection.");
  desc.addUntracked<std::string>("summaryTitle", "Summary")->setComment("Title of summary histogram.");
  desc.addUntracked<std::vector<int>>("ignoreBin", std::vector<int>())->setComment("List of bins to ignore");
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("l1tStage2MuonShowerComp", desc);
}

void L1TStage2MuonShowerComp::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {
  // Subsystem Monitoring and Muon Shower Output
  ibooker.setCurrentFolder(monitorDir_);

  summary_ = ibooker.book1D("summary",
                            summaryTitle_.c_str(),
                            numSummaryBins_,
                            1,
                            numSummaryBins_ + 1);  // range to match bin numbering
  summary_->setBinLabel(BXRANGEGOOD, "BX range match", 1);
  summary_->setBinLabel(BXRANGEBAD, "BX range mismatch", 1);
  summary_->setBinLabel(NSHOWERGOOD, "shower collection size match", 1);
  summary_->setBinLabel(NSHOWERBAD, "shower collection size mismatch", 1);
  summary_->setBinLabel(SHOWERALL, "# showers", 1);
  summary_->setBinLabel(SHOWERGOOD, "# matching showers", 1);
  summary_->setBinLabel(NOMINALBAD, "nominal shower mismatch", 1);
  summary_->setBinLabel(TIGHTBAD, "tight shower mismatch", 1);
  summary_->setBinLabel(LOOSEBAD, "loose shower mismatch", 1);

  errorSummaryNum_ = ibooker.book1D("errorSummaryNum",
                                    summaryTitle_.c_str(),
                                    numErrBins_,
                                    1,
                                    numErrBins_ + 1);  // range to match bin numbering
  errorSummaryNum_->setBinLabel(RBXRANGE, "BX range mismatch", 1);
  errorSummaryNum_->setBinLabel(RNSHOWER, "shower collection size mismatch", 1);
  errorSummaryNum_->setBinLabel(RSHOWER, "mismatching showers", 1);
  errorSummaryNum_->setBinLabel(RNOMINAL, "nominal shower mismatch", 1);
  errorSummaryNum_->setBinLabel(RTIGHT, "tight shower mismatch", 1);
  errorSummaryNum_->setBinLabel(RLOOSE, "loose shower mismatch", 1);

  // Change the label for those bins that will be ignored
  for (int i = 1; i <= errorSummaryNum_->getNbinsX(); i++) {
    if (incBin_[i] == false) {
      errorSummaryNum_->setBinLabel(i, "Ignored", 1);
    }
  }
  // Setting canExtend to false is needed to get the correct behaviour when running multithreaded.
  // Otherwise, when merging the histgrams of the threads, TH1::Merge sums bins that have the same label in one bin.
  // This needs to come after the calls to setBinLabel.
  errorSummaryNum_->getTH1F()->GetXaxis()->SetCanExtend(false);

  errorSummaryDen_ = ibooker.book1D(
      "errorSummaryDen", "denominators", numErrBins_, 1, numErrBins_ + 1);  // range to match bin numbering
  errorSummaryDen_->setBinLabel(RBXRANGE, "# events", 1);
  errorSummaryDen_->setBinLabel(RNSHOWER, "# shower collections", 1);
  for (int i = RSHOWER; i <= errorSummaryDen_->getNbinsX(); ++i) {
    errorSummaryDen_->setBinLabel(i, "# showers", 1);
  }
  // Needed for correct histogram summing in multithreaded running.
  errorSummaryDen_->getTH1F()->GetXaxis()->SetCanExtend(false);

  showerColl1BxRange_ =
      ibooker.book1D("showerBxRangeColl1", (showerColl1Title_ + " mismatching BX range").c_str(), 11, -5.5, 5.5);
  showerColl1BxRange_->setAxisTitle("BX range", 1);
  showerColl1nShowers_ =
      ibooker.book1D("nShowerColl1", (showerColl1Title_ + " mismatching shower multiplicity").c_str(), 37, -0.5, 36.5);
  showerColl1nShowers_->setAxisTitle("Shower multiplicity", 1);
  showerColl1ShowerTypeVsBX_ = ibooker.book2D("showerColl1ShowerTypeVsBX",
                                              showerColl1Title_ + " mismatching shower type occupancy per BX",
                                              7,
                                              -3.5,
                                              3.5,
                                              3,
                                              1,
                                              4);
  showerColl1ShowerTypeVsBX_->setAxisTitle("BX", 1);
  showerColl1ShowerTypeVsBX_->setAxisTitle("Shower type", 2);
  showerColl1ShowerTypeVsBX_->setBinLabel(IDX_LOOSE_SHOWER, "TwoLoose", 2);
  showerColl1ShowerTypeVsBX_->setBinLabel(IDX_TIGHT_SHOWER, "OneTight", 2);
  showerColl1ShowerTypeVsBX_->setBinLabel(IDX_NOMINAL_SHOWER, "OneNominal", 2);

  showerColl2BxRange_ =
      ibooker.book1D("showerBxRangeColl2", (showerColl2Title_ + " mismatching BX range").c_str(), 11, -5.5, 5.5);
  showerColl2BxRange_->setAxisTitle("BX range", 1);
  showerColl2nShowers_ =
      ibooker.book1D("nShowerColl2", (showerColl2Title_ + " mismatching shower multiplicity").c_str(), 37, -0.5, 36.5);
  showerColl2nShowers_->setAxisTitle("Shower multiplicity", 1);
  showerColl2ShowerTypeVsBX_ = ibooker.book2D("showerColl2ShowerTypeVsBX",
                                              showerColl2Title_ + " mismatching shower type occupancy per BX",
                                              7,
                                              -3.5,
                                              3.5,
                                              3,
                                              1,
                                              4);
  showerColl2ShowerTypeVsBX_->setAxisTitle("BX", 1);
  showerColl2ShowerTypeVsBX_->setAxisTitle("Shower type", 2);
  showerColl2ShowerTypeVsBX_->setBinLabel(IDX_LOOSE_SHOWER, "TwoLoose", 2);
  showerColl2ShowerTypeVsBX_->setBinLabel(IDX_TIGHT_SHOWER, "OneTight", 2);
  showerColl2ShowerTypeVsBX_->setBinLabel(IDX_NOMINAL_SHOWER, "OneNominal", 2);
}

void L1TStage2MuonShowerComp::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (verbose_) {
    edm::LogInfo("L1TStage2MuonShowerComp") << "L1TStage2MuonShowerComp: analyze..." << std::endl;
  }

  edm::Handle<l1t::MuonShowerBxCollection> showerBxColl1;
  edm::Handle<l1t::MuonShowerBxCollection> showerBxColl2;
  e.getByToken(showerToken1_, showerBxColl1);
  e.getByToken(showerToken2_, showerBxColl2);

  errorSummaryDen_->Fill(RBXRANGE);
  int bxRange1 = showerBxColl1->getLastBX() - showerBxColl1->getFirstBX() + 1;
  int bxRange2 = showerBxColl2->getLastBX() - showerBxColl2->getFirstBX() + 1;
  if (bxRange1 != bxRange2) {
    summary_->Fill(BXRANGEBAD);
    if (incBin_[RBXRANGE])
      errorSummaryNum_->Fill(RBXRANGE);
    int bx;
    for (bx = showerBxColl1->getFirstBX(); bx <= showerBxColl1->getLastBX(); ++bx) {
      showerColl1BxRange_->Fill(bx);
    }
    for (bx = showerBxColl2->getFirstBX(); bx <= showerBxColl2->getLastBX(); ++bx) {
      showerColl2BxRange_->Fill(bx);
    }
  } else {
    summary_->Fill(BXRANGEGOOD);
  }

  for (int iBx = showerBxColl1->getFirstBX(); iBx <= showerBxColl1->getLastBX(); ++iBx) {
    // don't analyse if this BX does not exist in the second collection
    if (iBx < showerBxColl2->getFirstBX() || iBx > showerBxColl2->getLastBX()) {
      continue;
    }

    l1t::MuonShowerBxCollection::const_iterator showerIt1;
    l1t::MuonShowerBxCollection::const_iterator showerIt2;

    errorSummaryDen_->Fill(RNSHOWER);
    // check number of showers
    if (showerBxColl1->size(iBx) != showerBxColl2->size(iBx)) {
      summary_->Fill(NSHOWERBAD);
      if (incBin_[RNSHOWER]) {
        errorSummaryNum_->Fill(RNSHOWER);
      }
      showerColl1nShowers_->Fill(showerBxColl1->size(iBx));
      showerColl2nShowers_->Fill(showerBxColl2->size(iBx));

      if (showerBxColl1->size(iBx) > showerBxColl2->size(iBx)) {
        showerIt1 = showerBxColl1->begin(iBx) + showerBxColl2->size(iBx);
        for (; showerIt1 != showerBxColl1->end(iBx); ++showerIt1) {
          if (showerIt1->isOneNominalInTime()) {
            showerColl1ShowerTypeVsBX_->Fill(IDX_NOMINAL_SHOWER, iBx);
          }
          if (showerIt1->isOneTightInTime()) {
            showerColl1ShowerTypeVsBX_->Fill(IDX_TIGHT_SHOWER, iBx);
          }
          if (showerIt1->isTwoLooseDiffSectorsInTime()) {
            showerColl1ShowerTypeVsBX_->Fill(IDX_LOOSE_SHOWER, iBx);
          }
        }
      } else {
        showerIt2 = showerBxColl2->begin(iBx) + showerBxColl1->size(iBx);
        for (; showerIt2 != showerBxColl2->end(iBx); ++showerIt2) {
          if (showerIt2->isOneNominalInTime()) {
            showerColl2ShowerTypeVsBX_->Fill(IDX_NOMINAL_SHOWER, iBx);
          }
          if (showerIt2->isOneTightInTime()) {
            showerColl2ShowerTypeVsBX_->Fill(IDX_TIGHT_SHOWER, iBx);
          }
          if (showerIt2->isTwoLooseDiffSectorsInTime()) {
            showerColl2ShowerTypeVsBX_->Fill(IDX_LOOSE_SHOWER, iBx);
          }
        }
      }
    } else {
      summary_->Fill(NSHOWERGOOD);
    }

    showerIt1 = showerBxColl1->begin(iBx);
    showerIt2 = showerBxColl2->begin(iBx);
    while (showerIt1 != showerBxColl1->end(iBx) && showerIt2 != showerBxColl2->end(iBx)) {
      summary_->Fill(SHOWERALL);
      for (int i = RSHOWER; i <= numErrBins_; ++i) {
        errorSummaryDen_->Fill(i);
      }

      bool showerMismatch = false;     // All shower mismatches
      bool showerSelMismatch = false;  // Muon mismatches excluding ignored bins
      if (showerIt1->isOneNominalInTime() != showerIt2->isOneNominalInTime()) {
        showerMismatch = true;
        summary_->Fill(NOMINALBAD);
        if (incBin_[RNOMINAL]) {
          showerSelMismatch = true;
          errorSummaryNum_->Fill(RNOMINAL);
        }
      }
      if (showerIt1->isOneTightInTime() != showerIt2->isOneTightInTime()) {
        showerMismatch = true;
        summary_->Fill(TIGHTBAD);
        if (incBin_[RTIGHT]) {
          showerSelMismatch = true;
          errorSummaryNum_->Fill(RTIGHT);
        }
      }

      if (incBin_[RSHOWER] && showerSelMismatch) {
        errorSummaryNum_->Fill(RSHOWER);
      }

      if (showerMismatch) {
        if (showerIt1->isOneNominalInTime()) {
          showerColl1ShowerTypeVsBX_->Fill(IDX_NOMINAL_SHOWER, iBx);
        }
        if (showerIt1->isOneTightInTime()) {
          showerColl1ShowerTypeVsBX_->Fill(IDX_TIGHT_SHOWER, iBx);
        }
        if (showerIt1->isTwoLooseDiffSectorsInTime()) {
          showerColl1ShowerTypeVsBX_->Fill(IDX_LOOSE_SHOWER, iBx);
        }

        if (showerIt2->isOneNominalInTime()) {
          showerColl2ShowerTypeVsBX_->Fill(IDX_NOMINAL_SHOWER, iBx);
        }
        if (showerIt2->isOneTightInTime()) {
          showerColl2ShowerTypeVsBX_->Fill(IDX_TIGHT_SHOWER, iBx);
        }
        if (showerIt2->isTwoLooseDiffSectorsInTime()) {
          showerColl2ShowerTypeVsBX_->Fill(IDX_LOOSE_SHOWER, iBx);
        }
      } else {
        summary_->Fill(SHOWERGOOD);
      }

      ++showerIt1;
      ++showerIt2;
    }
  }
}
