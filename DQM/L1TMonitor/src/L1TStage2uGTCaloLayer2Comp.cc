#include "DQM/L1TMonitor/interface/L1TStage2uGTCaloLayer2Comp.h"

L1TStage2uGTCaloLayer2Comp::L1TStage2uGTCaloLayer2Comp (const edm::ParameterSet& ps)
  : monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
    calol2JetCollection(consumes <l1t::JetBxCollection>(
			  ps.getParameter<edm::InputTag>(
			    "calol2JetCollection"))),
    uGTJetCollection(consumes <l1t::JetBxCollection>(
		       ps.getParameter<edm::InputTag>(
			 "uGTJetCollection"))),
    calol2EGammaCollection(consumes <l1t::EGammaBxCollection>(
			     ps.getParameter<edm::InputTag>(
			       "calol2EGammaCollection"))),
    uGTEGammaCollection(consumes <l1t::EGammaBxCollection>(
			  ps.getParameter<edm::InputTag>(
			    "uGTEGammaCollection"))),
    calol2TauCollection(consumes <l1t::TauBxCollection>(
			  ps.getParameter<edm::InputTag>(
			    "calol2TauCollection"))),
    uGTTauCollection(consumes <l1t::TauBxCollection>(
		       ps.getParameter<edm::InputTag>(
			 "uGTTauCollection"))),
    calol2EtSumCollection(consumes <l1t::EtSumBxCollection>(
			    ps.getParameter<edm::InputTag>(
			      "calol2EtSumCollection"))),
    uGTEtSumCollection(consumes <l1t::EtSumBxCollection>(
			 ps.getParameter<edm::InputTag>(
			   "uGTEtSumCollection"))),
    verbose(ps.getUntrackedParameter<bool> ("verbose", false))
{}

void L1TStage2uGTCaloLayer2Comp::bookHistograms(
  DQMStore::IBooker &ibooker,
  edm::Run const &,
  edm::EventSetup const&) {

  ibooker.setCurrentFolder(monitorDir);

  // the index of the first bin in histogram should match value of first enum
  comparisonNum = ibooker.book1D(
    "errorSummaryNum",
    "CaloLayer2-uGT link check - Numerator (# disagreements)", 15, 1, 16);

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

  comparisonDenum = ibooker.book1D(
    "errorSummaryDen",
    "CaloLayer2-uGT link check - Denumerator (# objects)", 15, 1, 16);

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
void L1TStage2uGTCaloLayer2Comp::analyze (
  const edm::Event& e,
  const edm::EventSetup & c) {

  // define collections to hold lists of objects in event
  edm::Handle<l1t::JetBxCollection> jetColCalol2;
  edm::Handle<l1t::JetBxCollection> jetColuGT;
  edm::Handle<l1t::EGammaBxCollection> egColCalol2;
  edm::Handle<l1t::EGammaBxCollection> egColuGT;
  edm::Handle<l1t::TauBxCollection> tauColCalol2;
  edm::Handle<l1t::TauBxCollection> tauColuGT;
  edm::Handle<l1t::EtSumBxCollection> sumColCalol2;
  edm::Handle<l1t::EtSumBxCollection> sumColuGT;

  // map event contents to above collections
  e.getByToken(calol2JetCollection, jetColCalol2);
  e.getByToken(uGTJetCollection, jetColuGT);
  e.getByToken(calol2EGammaCollection, egColCalol2);
  e.getByToken(uGTEGammaCollection, egColuGT);
  e.getByToken(calol2TauCollection, tauColCalol2);
  e.getByToken(uGTTauCollection, tauColuGT);
  e.getByToken(calol2EtSumCollection, sumColCalol2);
  e.getByToken(uGTEtSumCollection, sumColuGT);

  bool eventGood = true;

  if (!compareJets(jetColCalol2, jetColuGT)) {
    eventGood = false;
  }

  if (!compareEGs(egColCalol2, egColuGT)) {
    eventGood = false;
  }

  if (!compareTaus(tauColCalol2, tauColuGT)) {
    eventGood = false;
  }

  if (!compareSums(sumColCalol2, sumColuGT)) {
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
bool L1TStage2uGTCaloLayer2Comp::compareJets(
  const edm::Handle<l1t::JetBxCollection> & calol2Col,
  const edm::Handle<l1t::JetBxCollection> & uGTCol)
{
  bool eventGood = true;

  l1t::JetBxCollection::const_iterator calol2It = calol2Col->begin();
  l1t::JetBxCollection::const_iterator uGTIt = uGTCol->begin();

  // process jets
  if (calol2Col->size() != uGTCol->size()) {
    comparisonNum->Fill(EVENTBADJETCOL);
    return false;
  }

  int nJets = 0;
  if (calol2It != calol2Col->end() ||
      uGTIt != uGTCol->end()) {
    while(true) {

      ++nJets;

      // object pt mismatch
      if (calol2It->hwPt() != uGTIt->hwPt()) {
	comparisonNum->Fill(JETBADET);
 	eventGood = false;
      }

      // object position mismatch (phi)
      if (calol2It->hwPhi() != uGTIt->hwPhi()){
	comparisonNum->Fill(JETBADPHI);
	eventGood = false;
      }

      // object position mismatch (eta)
      if (calol2It->hwEta() != uGTIt->hwEta()) {
	comparisonNum->Fill(JETBADETA);
	eventGood = false;
      }

      // keep track of jets
      comparisonDenum->Fill(JETS1);
      comparisonDenum->Fill(JETS2);
      comparisonDenum->Fill(JETS3);

      // increment position of pointers
      ++calol2It;
      ++uGTIt;

      if (calol2It == calol2Col->end() ||
	  uGTIt == uGTCol->end())
	break;
    }
  } else {
    if (calol2Col->size() != 0 || uGTCol->size() != 0) {
      comparisonNum->Fill(EVENTBADJETCOL);
      return false;
    }
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for e/gammas
bool L1TStage2uGTCaloLayer2Comp::compareEGs(
  const edm::Handle<l1t::EGammaBxCollection> & calol2Col,
  const edm::Handle<l1t::EGammaBxCollection> & uGTCol)
{
  bool eventGood = true;

  l1t::EGammaBxCollection::const_iterator calol2It = calol2Col->begin();
  l1t::EGammaBxCollection::const_iterator uGTIt = uGTCol->begin();

  // check length of collections
  if (calol2Col->size() != uGTCol->size()) {
    comparisonNum->Fill(EVENTBADEGCOL);
    return false;
  }

  // processing continues only of length of object collections is the same
  if (calol2It != calol2Col->end() ||
      uGTIt != uGTCol->end()) {

    while(true) {

      // object pt mismatch
      if (calol2It->hwPt() != uGTIt->hwPt()) {
	comparisonNum->Fill(EGBADET);
	eventGood = false;
      }

      // object position mismatch (phi)
      if (calol2It->hwPhi() != uGTIt->hwPhi()) {
	comparisonNum->Fill(EGBADPHI);
	eventGood = false;
      }

      // object position mismatch (eta)
      if (calol2It->hwEta() != uGTIt->hwEta()) {
	comparisonNum->Fill(EGBADETA);
	eventGood = false;
      }

      // keep track of number of objects
      comparisonDenum->Fill(EGS1);
      comparisonDenum->Fill(EGS2);
      comparisonDenum->Fill(EGS3);

      // increment position of pointers
      ++calol2It;
      ++uGTIt;

      if (calol2It == calol2Col->end() ||
	  uGTIt == uGTCol->end())
	break;
    }
  } else {
    if (calol2Col->size() != 0 || uGTCol->size() != 0) {
      comparisonNum->Fill(EVENTBADEGCOL);
      return false;
    }
  }

  // return a boolean that states whether the eg data in the event is in
  // agreement
  return eventGood;
}

// comparison method for taus
bool L1TStage2uGTCaloLayer2Comp::compareTaus(
  const edm::Handle<l1t::TauBxCollection> & calol2Col,
  const edm::Handle<l1t::TauBxCollection> & uGTCol)
{
  bool eventGood = true;

  l1t::TauBxCollection::const_iterator calol2It = calol2Col->begin();
  l1t::TauBxCollection::const_iterator uGTIt = uGTCol->begin();

  // check length of collections
  if (calol2Col->size() != uGTCol->size()) {
    comparisonNum->Fill(EVENTBADTAUCOL);
    return false;
  }

  // processing continues only of length of object collections is the same
  if (calol2It != calol2Col->end() ||
      uGTIt != uGTCol->end()) {

    while(true) {
      // object Et mismatch
      if (calol2It->hwPt() != uGTIt->hwPt()) {
	comparisonNum->Fill(TAUBADET);
	eventGood = false;
      }

      // object position mismatch (phi)
      if (calol2It->hwPhi() != uGTIt->hwPhi()) {
	comparisonNum->Fill(TAUBADPHI);
	eventGood = false;
      }

      // object position mismatch (eta)
      if (calol2It->hwEta() != uGTIt->hwEta()) {
	comparisonNum->Fill(TAUBADETA);
	eventGood = false;
      }

      // keep track of number of objects
      comparisonDenum->Fill(TAUS1);
      comparisonDenum->Fill(TAUS2);
      comparisonDenum->Fill(TAUS3);

      // increment position of pointers
      ++calol2It;
      ++uGTIt;

      if (calol2It == calol2Col->end() ||
	  uGTIt == uGTCol->end())
	break;
    }
  } else {
    if (calol2Col->size() != 0 || uGTCol->size() != 0) {
      comparisonNum->Fill(EVENTBADTAUCOL);
      return false;
    }
  }

  // return a boolean that states whether the tau data in the event is in
  // agreement
  return eventGood;
}

// comparison method for sums
bool L1TStage2uGTCaloLayer2Comp::compareSums(
  const edm::Handle<l1t::EtSumBxCollection> & calol2Col,
  const edm::Handle<l1t::EtSumBxCollection> & uGTCol)
{
  bool eventGood = true;

  double calol2Et  = 0;
  double uGTEt     = 0;
  double calol2Phi = 0;
  double uGTPhi    = 0;

  // if the calol2 or ugt collections have different size, mark the event as
  // bad (this should never occur in normal running)
  if (calol2Col->size() != uGTCol->size()) {
    comparisonNum->Fill(EVENTBADSUMCOL);
    return false;
  }

  l1t::EtSumBxCollection::const_iterator calol2It = calol2Col->begin();
  l1t::EtSumBxCollection::const_iterator uGTIt = uGTCol->begin();

  while (calol2It != calol2Col->end() && uGTIt != uGTCol->end()) {

    // ETT, ETTEM, HTT, TowCnt, MBHFP0, MBHFM0, MBHFP1 or MBHFM1
    if ((l1t::EtSum::EtSumType::kTotalEt == calol2It->getType()) ||    // ETT
	(l1t::EtSum::EtSumType::kTotalEtEm == calol2It->getType()) ||  // ETTEM
	(l1t::EtSum::EtSumType::kTotalHt == calol2It->getType()) ||    // HTT
	(l1t::EtSum::EtSumType::kTowerCount == calol2It->getType()) || // TowCnt
    	(l1t::EtSum::EtSumType::kMinBiasHFP0 == calol2It->getType()) ||// MBHFP0
	(l1t::EtSum::EtSumType::kMinBiasHFM0 == calol2It->getType()) ||// MBHFM0
	(l1t::EtSum::EtSumType::kMinBiasHFP1 == calol2It->getType()) ||// MBHFP1
	(l1t::EtSum::EtSumType::kMinBiasHFM1 == calol2It->getType())) {// MBHFM1

      calol2Et = calol2It->hwPt();
      uGTEt = uGTIt->hwPt();

      if (calol2Et != uGTEt) {
	eventGood = false;
	comparisonNum->Fill(BADSUM);
      }

      // update sum counters
      comparisonDenum->Fill(SUMS);
    }

    // MET, METHF, MHT or MHTHF
    if ((l1t::EtSum::EtSumType::kMissingEt == calol2It->getType()) ||   // MET
	(l1t::EtSum::EtSumType::kMissingEtHF == calol2It->getType()) || // METHF
	(l1t::EtSum::EtSumType::kMissingHt == calol2It->getType()) ||   // MHT
	(l1t::EtSum::EtSumType::kMissingHtHF == calol2It->getType())) { // MHTHF

      calol2Et = calol2It->hwPt();
      uGTEt = uGTIt->hwPt();

      calol2Phi = calol2It->hwPhi();
      uGTPhi = uGTIt->hwPhi();

      if ((calol2Et != uGTEt) || (calol2Phi != uGTPhi)) {
	eventGood = false;
	comparisonNum->Fill(BADSUM);
      }

      // update sum counters
      comparisonDenum->Fill(SUMS);
    }

    ++calol2It;
    ++uGTIt;
  }

  // return a boolean that states whether the sum data in the event is in
  // agreement
  return eventGood;
}


