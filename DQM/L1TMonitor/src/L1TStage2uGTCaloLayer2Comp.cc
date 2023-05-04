#include "DQM/L1TMonitor/interface/L1TStage2uGTCaloLayer2Comp.h"

L1TStage2uGTCaloLayer2Comp::L1TStage2uGTCaloLayer2Comp(const edm::ParameterSet& ps)
    : monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      collection1Title(ps.getUntrackedParameter<std::string>("collection1Title")),
      collection2Title(ps.getUntrackedParameter<std::string>("collection2Title")),
      JetCollection1(consumes<l1t::JetBxCollection>(ps.getParameter<edm::InputTag>("JetCollection1"))),
      JetCollection2(consumes<l1t::JetBxCollection>(ps.getParameter<edm::InputTag>("JetCollection2"))),
      EGammaCollection1(consumes<l1t::EGammaBxCollection>(ps.getParameter<edm::InputTag>("EGammaCollection1"))),
      EGammaCollection2(consumes<l1t::EGammaBxCollection>(ps.getParameter<edm::InputTag>("EGammaCollection2"))),
      TauCollection1(consumes<l1t::TauBxCollection>(ps.getParameter<edm::InputTag>("TauCollection1"))),
      TauCollection2(consumes<l1t::TauBxCollection>(ps.getParameter<edm::InputTag>("TauCollection2"))),
      EtSumCollection1(consumes<l1t::EtSumBxCollection>(ps.getParameter<edm::InputTag>("EtSumCollection1"))),
      EtSumCollection2(consumes<l1t::EtSumBxCollection>(ps.getParameter<edm::InputTag>("EtSumCollection2"))),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

void L1TStage2uGTCaloLayer2Comp::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(monitorDir);

  // the index of the first bin in histogram should match value of first enum
  comparisonNum =
      ibooker.book1D("errorSummaryNum",
                     collection1Title + " vs " + collection2Title + " Comparison - Numerator (# Disagreements)",
                     15,
                     1,
                     16);

  comparisonNum->setBinLabel(EVENTBAD, "# bad evts");
  comparisonNum->setBinLabel(EVENTBADJETCOL, "# evts w/ bad jet col size");
  comparisonNum->setBinLabel(EVENTBADEGCOL, "# evts w/ bad eg col size");
  comparisonNum->setBinLabel(EVENTBADTAUCOL, "# evts w/ bad tau col size");
  comparisonNum->setBinLabel(EVENTBADSUMCOL, "# evts w/ bad sum col size");
  comparisonNum->setBinLabel(JETBADET, "# jets bad Et");
  comparisonNum->setBinLabel(JETBADPHI, "# jets bad phi");
  comparisonNum->setBinLabel(JETBADETA, "# jets bad eta");
  comparisonNum->setBinLabel(EGBADET, "# egs bad Et");
  comparisonNum->setBinLabel(EGBADPHI, "# egs bad phi");
  comparisonNum->setBinLabel(EGBADETA, "# egs bad eta");
  comparisonNum->setBinLabel(TAUBADET, "# taus bad Et");
  comparisonNum->setBinLabel(TAUBADPHI, "# taus bad phi");
  comparisonNum->setBinLabel(TAUBADETA, "# taus bad eta");
  comparisonNum->setBinLabel(BADSUM, "# bad sums");

  comparisonDenum =
      ibooker.book1D("errorSummaryDen",
                     collection1Title + " vs " + collection2Title + " Comparison - Denominator (# Objects)",
                     15,
                     1,
                     16);

  comparisonDenum->setBinLabel(EVENTS1, "# evts");
  comparisonDenum->setBinLabel(EVENTS2, "# evts");
  comparisonDenum->setBinLabel(EVENTS3, "# evts");
  comparisonDenum->setBinLabel(EVENTS4, "# evts");
  comparisonDenum->setBinLabel(EVENTS5, "# evts");
  comparisonDenum->setBinLabel(JETS1, "# jets");
  comparisonDenum->setBinLabel(JETS2, "# jets");
  comparisonDenum->setBinLabel(JETS3, "# jets");
  comparisonDenum->setBinLabel(EGS1, "# egs");
  comparisonDenum->setBinLabel(EGS2, "# egs");
  comparisonDenum->setBinLabel(EGS3, "# egs");
  comparisonDenum->setBinLabel(TAUS1, "# taus");
  comparisonDenum->setBinLabel(TAUS2, "# taus");
  comparisonDenum->setBinLabel(TAUS3, "# taus");
  comparisonDenum->setBinLabel(SUMS, "# sums");
  // Setting canExtend to false is needed to get the correct behaviour when running multithreaded.
  // Otherwise, when merging the histgrams of the threads, TH1::Merge sums bins that have the same label in one bin.
  // This needs to come after the calls to setBinLabel.
  comparisonDenum->getTH1F()->GetXaxis()->SetCanExtend(false);
}
void L1TStage2uGTCaloLayer2Comp::analyze(const edm::Event& e, const edm::EventSetup& c) {
  // define collections to hold lists of objects in event
  edm::Handle<l1t::JetBxCollection> jetCol1;
  edm::Handle<l1t::JetBxCollection> jetCol2;
  edm::Handle<l1t::EGammaBxCollection> egCol1;
  edm::Handle<l1t::EGammaBxCollection> egCol2;
  edm::Handle<l1t::TauBxCollection> tauCol1;
  edm::Handle<l1t::TauBxCollection> tauCol2;
  edm::Handle<l1t::EtSumBxCollection> sumCol1;
  edm::Handle<l1t::EtSumBxCollection> sumCol2;

  // map event contents to above collections
  e.getByToken(JetCollection1, jetCol1);
  e.getByToken(JetCollection2, jetCol2);
  e.getByToken(EGammaCollection1, egCol1);
  e.getByToken(EGammaCollection2, egCol2);
  e.getByToken(TauCollection1, tauCol1);
  e.getByToken(TauCollection2, tauCol2);
  e.getByToken(EtSumCollection1, sumCol1);
  e.getByToken(EtSumCollection2, sumCol2);

  bool eventGood = true;

  if (!compareJets(jetCol1, jetCol2)) {
    eventGood = false;
  }

  if (!compareEGs(egCol1, egCol2)) {
    eventGood = false;
  }

  if (!compareTaus(tauCol1, tauCol2)) {
    eventGood = false;
  }

  if (!compareSums(sumCol1, sumCol2)) {
    eventGood = false;
  }

  if (!eventGood) {
    comparisonNum->Fill(EVENTBAD);
  }

  comparisonDenum->Fill(EVENTS1);
  comparisonDenum->Fill(EVENTS2);
  comparisonDenum->Fill(EVENTS3);
  comparisonDenum->Fill(EVENTS4);
  comparisonDenum->Fill(EVENTS5);
}

// comparison method for jets
bool L1TStage2uGTCaloLayer2Comp::compareJets(const edm::Handle<l1t::JetBxCollection>& col1,
                                             const edm::Handle<l1t::JetBxCollection>& col2) {
  bool eventGood = true;

  l1t::JetBxCollection::const_iterator col1It = col1->begin();
  l1t::JetBxCollection::const_iterator col2It = col2->begin();

  // process jets
  if (col1->size() != col2->size()) {
    comparisonNum->Fill(EVENTBADJETCOL);
    return false;
  }

  if (col1It != col1->end() || col2It != col2->end()) {
    while (true) {

      // object pt mismatch
      if (col1It->hwPt() != col2It->hwPt()) {
        comparisonNum->Fill(JETBADET);
        eventGood = false;
      }

      // object position mismatch (phi)
      if (col1It->hwPhi() != col2It->hwPhi()) {
        comparisonNum->Fill(JETBADPHI);
        eventGood = false;
      }

      // object position mismatch (eta)
      if (col1It->hwEta() != col2It->hwEta()) {
        comparisonNum->Fill(JETBADETA);
        eventGood = false;
      }

      // keep track of jets
      comparisonDenum->Fill(JETS1);
      comparisonDenum->Fill(JETS2);
      comparisonDenum->Fill(JETS3);

      // increment position of pointers
      ++col1It;
      ++col2It;

      if (col1It == col1->end() || col2It == col2->end())
        break;
    }
  } else {
    if (col1->size() != 0 || col2->size() != 0) {
      comparisonNum->Fill(EVENTBADJETCOL);
      return false;
    }
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for e/gammas
bool L1TStage2uGTCaloLayer2Comp::compareEGs(const edm::Handle<l1t::EGammaBxCollection>& col1,
                                            const edm::Handle<l1t::EGammaBxCollection>& col2) {
  bool eventGood = true;

  l1t::EGammaBxCollection::const_iterator col1It = col1->begin();
  l1t::EGammaBxCollection::const_iterator col2It = col2->begin();

  // check length of collections
  if (col1->size() != col2->size()) {
    comparisonNum->Fill(EVENTBADEGCOL);
    return false;
  }

  // processing continues only of length of object collections is the same
  if (col1It != col1->end() || col2It != col2->end()) {
    while (true) {
      // object pt mismatch
      if (col1It->hwPt() != col2It->hwPt()) {
        comparisonNum->Fill(EGBADET);
        eventGood = false;
      }

      // object position mismatch (phi)
      if (col1It->hwPhi() != col2It->hwPhi()) {
        comparisonNum->Fill(EGBADPHI);
        eventGood = false;
      }

      // object position mismatch (eta)
      if (col1It->hwEta() != col2It->hwEta()) {
        comparisonNum->Fill(EGBADETA);
        eventGood = false;
      }

      // keep track of number of objects
      comparisonDenum->Fill(EGS1);
      comparisonDenum->Fill(EGS2);
      comparisonDenum->Fill(EGS3);

      // increment position of pointers
      ++col1It;
      ++col2It;

      if (col1It == col1->end() || col2It == col2->end())
        break;
    }
  } else {
    if (col1->size() != 0 || col2->size() != 0) {
      comparisonNum->Fill(EVENTBADEGCOL);
      return false;
    }
  }

  // return a boolean that states whether the eg data in the event is in
  // agreement
  return eventGood;
}

// comparison method for taus
bool L1TStage2uGTCaloLayer2Comp::compareTaus(const edm::Handle<l1t::TauBxCollection>& col1,
                                             const edm::Handle<l1t::TauBxCollection>& col2) {
  bool eventGood = true;

  l1t::TauBxCollection::const_iterator col1It = col1->begin();
  l1t::TauBxCollection::const_iterator col2It = col2->begin();

  // check length of collections
  if (col1->size() != col2->size()) {
    comparisonNum->Fill(EVENTBADTAUCOL);
    return false;
  }

  // processing continues only of length of object collections is the same
  if (col1It != col1->end() || col2It != col2->end()) {
    while (true) {
      // object Et mismatch
      if (col1It->hwPt() != col2It->hwPt()) {
        comparisonNum->Fill(TAUBADET);
        eventGood = false;
      }

      // object position mismatch (phi)
      if (col1It->hwPhi() != col2It->hwPhi()) {
        comparisonNum->Fill(TAUBADPHI);
        eventGood = false;
      }

      // object position mismatch (eta)
      if (col1It->hwEta() != col2It->hwEta()) {
        comparisonNum->Fill(TAUBADETA);
        eventGood = false;
      }

      // keep track of number of objects
      comparisonDenum->Fill(TAUS1);
      comparisonDenum->Fill(TAUS2);
      comparisonDenum->Fill(TAUS3);

      // increment position of pointers
      ++col1It;
      ++col2It;

      if (col1It == col1->end() || col2It == col2->end())
        break;
    }
  } else {
    if (col1->size() != 0 || col2->size() != 0) {
      comparisonNum->Fill(EVENTBADTAUCOL);
      return false;
    }
  }

  // return a boolean that states whether the tau data in the event is in
  // agreement
  return eventGood;
}

// comparison method for sums
bool L1TStage2uGTCaloLayer2Comp::compareSums(const edm::Handle<l1t::EtSumBxCollection>& col1,
                                             const edm::Handle<l1t::EtSumBxCollection>& col2) {
  bool eventGood = true;

  double col1Et = 0;
  double col2Et = 0;
  double col1Phi = 0;
  double col2Phi = 0;

  // if the calol2 or ugt collections have different size, mark the event as
  // bad (this should never occur in normal running)
  if (col1->size() != col2->size()) {
    comparisonNum->Fill(EVENTBADSUMCOL);
    return false;
  }

  l1t::EtSumBxCollection::const_iterator col1It = col1->begin();
  l1t::EtSumBxCollection::const_iterator col2It = col2->begin();

  while (col1It != col1->end() && col2It != col2->end()) {
    // ETT, ETTEM, HTT, TowCnt, MBHFP0, MBHFM0, MBHFP1 or MBHFM1
    if ((l1t::EtSum::EtSumType::kTotalEt == col1It->getType()) ||      // ETT
        (l1t::EtSum::EtSumType::kTotalEtEm == col1It->getType()) ||    // ETTEM
        (l1t::EtSum::EtSumType::kTotalHt == col1It->getType()) ||      // HTT
        (l1t::EtSum::EtSumType::kTowerCount == col1It->getType()) ||   // TowCnt
        (l1t::EtSum::EtSumType::kMinBiasHFP0 == col1It->getType()) ||  // MBHFP0
        (l1t::EtSum::EtSumType::kMinBiasHFM0 == col1It->getType()) ||  // MBHFM0
        (l1t::EtSum::EtSumType::kMinBiasHFP1 == col1It->getType()) ||  // MBHFP1
        (l1t::EtSum::EtSumType::kMinBiasHFM1 == col1It->getType())) {  // MBHFM1

      col1Et = col1It->hwPt();
      col2Et = col2It->hwPt();

      if (col1Et != col2Et) {
        eventGood = false;
        comparisonNum->Fill(BADSUM);
      }

      // update sum counters
      comparisonDenum->Fill(SUMS);
    }

    // MET, METHF, MHT or MHTHF
    if ((l1t::EtSum::EtSumType::kMissingEt == col1It->getType()) ||    // MET
        (l1t::EtSum::EtSumType::kMissingEtHF == col1It->getType()) ||  // METHF
        (l1t::EtSum::EtSumType::kMissingHt == col1It->getType()) ||    // MHT
        (l1t::EtSum::EtSumType::kMissingHtHF == col1It->getType())) {  // MHTHF

      col1Et = col1It->hwPt();
      col2Et = col2It->hwPt();

      col1Phi = col1It->hwPhi();
      col2Phi = col2It->hwPhi();

      if ((col1Et != col2Et) || (col1Phi != col2Phi)) {
        eventGood = false;
        comparisonNum->Fill(BADSUM);
      }

      // update sum counters
      comparisonDenum->Fill(SUMS);
    }

    ++col1It;
    ++col2It;
  }

  // return a boolean that states whether the sum data in the event is in
  // agreement
  return eventGood;
}
