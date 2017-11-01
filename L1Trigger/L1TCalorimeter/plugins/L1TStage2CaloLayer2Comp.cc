#ifndef L1Trigger_L1TCalorimeter_L1TStage2CaloLayer2Comp
#define L1Trigger_L1TCalorimeter_L1TStage2CaloLayer2Comp

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// #include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

// classes required for storage of objects into a separate edm file
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

// classes required for sorting of jets
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/AccumulatingSort.h"
#include "L1Trigger/L1TCalorimeter/interface/BitonicSort.h"
#include "DataFormats/Math/interface/LorentzVector.h"


namespace l1t {
  bool operator > ( const l1t::Jet& a, l1t::Jet& b ) {
    return  a.hwPt() > b.hwPt();
  }
  bool operator > ( const l1t::EGamma& a, l1t::EGamma& b ) {
    return  a.hwPt() > b.hwPt();
  }
  bool operator > ( const l1t::Tau& a, l1t::Tau& b ) {
    return  a.hwPt() > b.hwPt();
  }
}


#include "TH1F.h"
#include "TH2F.h"


/**
 * Short class description.
 *
 * Longer class description...
 * ... desc continued.
 */
class  L1TStage2CaloLayer2Comp : public edm::EDProducer {

public:
  /**
   * Class constructor
   *
   * Receives the set of parameters, specified by the python configuration file
   * used to initialise the module as a part of a sequence. The contents of the
   * set is used to configure the internal state of the objects of this class.
   * Values from the parameter set are extracted and used to initialise
   * bxcollections for jet, e/g, tau and sum objects reconstructed by firmware
   * (data) and emulator. These collections are the basis of the comparisons
   * performed by this module.
   *
   * @param edm::ParamterSet & ps A pointer to the parameter set used
   */
  L1TStage2CaloLayer2Comp (const edm::ParameterSet & ps);


protected:

  /**
   * Main method where the analysis code resides, executed once for each run
   *
   * The main body of the module code is contained in this method. The different
   * object collections are extracted from the run and are passed to the
   * respective comparison methods for processing of each object type.
   *
   * @param edm::Event const &         Reference to event object
   * @param edm::EventSetup const &    Reference to event configuration object
   *
   * @return void
   */
  virtual void produce (edm::Event&, const edm::EventSetup&);

private:

  /**
   * Encapsulates the code required for performing a comparison of
   * the jets contained in a given event.
   *
   * Method is called once per each event with the jet collections associated
   * with the event being extracted for bx = 0 as the RAW information required
   * to run the emulator is only available for bx 0. The implementation checks
   * if the size of collections is the same and when so, compares the jets in
   * the same positions within the data/emul collections. When a discrepancy is
   * found, the properties (Et, eta, phi) of the problematic data/emul objects
   * are stored in dedicated histograms. In cases of differences, a distinction
   * is made between objects whose energy or position are in disagremeent.When
   * the length of data/emul collections is different, all of the objects are
   * "stored" in the histograms dedicated to problematic objects.
   *
   * @param edm::Handle<l1t::JetBXCollection>& dataCol Reference to jet
   *    collection from data
   * @param edm::Handle<l1t::JetBXCollection>& emulCol Reference to jet
   *    collection from emulation
   *
   * @return bool Flag of whether the agreement was perfect
   */
  bool compareJets(const edm::Handle<l1t::JetBxCollection> & dataCol,
                   const edm::Handle<l1t::JetBxCollection> & emulCol);

  /**
   * Encapsulates the code required for performing a comparison of
   * the e/gs contained in a given event.
   *
   * Method is called once per each event with the e/g collections associated
   * with the event being extracted for bx = 0 as the RAW information required
   * to run the emulator is only available for bx 0. The implementation checks
   * if the size of collections is the same and when so, compares the e/gs in
   * the same positions within the data/emul collections. When a discrepancy is
   * found, the properties (Et, eta, phi) of the problematic data/emul objects
   * are stored in dedicated histograms. In cases of differences, a distinction
   * is made between objects whose energy or position are in disagremeent.
   * Another distinction is made between isolated and non-isolated e/g
   * candidates and problematic objects are handled accordingly. When the length
   * of data/emul collections is different, all of the objects are "stored" in
   * the histograms dedicated to problematic objects.
   *
   * @param edm::Handle<l1t::EGammaBXCollection>& dataCol Reference to e/gamma
   *    collection from data
   * @param edm::Handle<l1t::EGammaBXCollection>& emulCol Reference to e/gamma
   *    collection from emulation
   *
   * @return bool Flag of whether the agreement was perfect
   */
  bool compareEGs(const edm::Handle<l1t::EGammaBxCollection> & dataCol,
                  const edm::Handle<l1t::EGammaBxCollection> & emulCol);

  /**
   * Encapsulates the code required for performing a comparison of
   * the taus contained in a given event.
   *
   * Method is called once per each event with the tau collections associated
   * with the event being extracted for bx = 0 as the RAW information required
   * to run the emulator is only available for bx 0. The implementation checks
   * if the size of collections is the same and when so, compares the taus in
   * the same positions within the data/emul collections. When a discrepancy is
   * found, the properties (Et, eta, phi) of the problematic data/emul objects
   * are stored in dedicated histograms. In cases of differences, a distinction
   * is made between objects whose energy or position are in disagremeent.
   * Another distinction is made between isolated and non-isolated tau
   * candidates and problematic objects are handled accordingly. When the length
   * of data/emul collections is different, all of the objects are "stored" in
   * the histograms dedicated to problematic objects.
   *
   * @param edm::Handle<l1t::TauBXCollection>& dataCol Reference to tau
   *    collection from data
   * @param edm::Handle<l1t::TauBXCollection>& emulCol Reference to tau
   *    collection from emulation
   *
   * @return bool Flag of whether the agreement was perfect
   */
  bool compareTaus(const edm::Handle<l1t::TauBxCollection> & dataCol,
                   const edm::Handle<l1t::TauBxCollection> & emulCol);

  /**
   * Encapsulates the code required for performing a comparison of
   * the taus contained in a given event.
   *
   * Method is called once per each event with the sum collections associated
   * with the event being extracted for bx = 0 as the RAW information required
   * to run the emulator is only available for bx 0. The implementation loops
   * over the collection and depending of the sum type, each variant is compared
   * independently. If any disagreement is found, the event is marked a bad and
   * the properties of the sum are stored in the data/emulator problematic
   * histograms.
   *
   * @param edm::Handle<l1t::TauBXCollection>& dataCol Reference to tau
   *    collection from data
   * @param edm::Handle<l1t::TauBXCollection>& emulCol Reference to tau
   *    collection from emulation
   *
   * @return bool Flag of whether the agreement was perfect
   */
  bool compareSums(const edm::Handle<l1t::EtSumBxCollection> & dataCol,
                   const edm::Handle<l1t::EtSumBxCollection> & emulCol);

  void accuSort(std::vector<l1t::Jet> & jets);
  void accuSort(std::vector<l1t::Tau> & taus);
  void accuSort(std::vector<l1t::EGamma> & egs);
  void accuSort(std::vector<l1t::L1Candidate> & jets);

  void dumpEventToFile();
  void dumpEventToEDM(edm::Event &e);

  // mapping between sums in emulator and data
  int emul_to_data_sum_index_map[31] = {
    9, 1, 19, 8, 0, 18, 10, 4, 6, 14,     // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    28, 24, 13, 27, 23, 15, 5, 7, 22, 12, // 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    3, 21, 11, 2, 20, 17, 30, 26, 16, 29, // 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
    25                                    // 30, 31
  };

  // collections to hold entities reconstructed from data and emulation
  edm::EDGetToken calol2JetCollectionData;
  edm::EDGetToken calol2JetCollectionEmul;
  edm::EDGetToken calol2EGammaCollectionData;
  edm::EDGetToken calol2EGammaCollectionEmul;
  edm::EDGetToken calol2TauCollectionData;
  edm::EDGetToken calol2TauCollectionEmul;
  edm::EDGetToken calol2EtSumCollectionData;
  edm::EDGetToken calol2EtSumCollectionEmul;
  edm::EDGetToken calol2CaloTowerCollectionData;
  edm::EDGetToken calol2CaloTowerCollectionEmul;

  // define collections to hold lists of objects in event
  edm::Handle<l1t::JetBxCollection> jetDataCol;
  edm::Handle<l1t::JetBxCollection> jetEmulCol;
  edm::Handle<l1t::EGammaBxCollection> egDataCol;
  edm::Handle<l1t::EGammaBxCollection> egEmulCol;
  edm::Handle<l1t::TauBxCollection> tauDataCol;
  edm::Handle<l1t::TauBxCollection> tauEmulCol;
  edm::Handle<l1t::EtSumBxCollection> sumDataCol;
  edm::Handle<l1t::EtSumBxCollection> sumEmulCol;
  edm::Handle<l1t::CaloTowerBxCollection> caloTowerDataCol;
  edm::Handle<l1t::CaloTowerBxCollection> caloTowerEmulCol;

  const unsigned int currBx = 0;
};

L1TStage2CaloLayer2Comp::L1TStage2CaloLayer2Comp (const edm::ParameterSet& ps)
  : calol2JetCollectionData(consumes <l1t::JetBxCollection>(
			      ps.getParameter<edm::InputTag>(
				"calol2JetCollectionData"))),
    calol2JetCollectionEmul(consumes <l1t::JetBxCollection>(
			      ps.getParameter<edm::InputTag>(
				"calol2JetCollectionEmul"))),
    calol2EGammaCollectionData(consumes <l1t::EGammaBxCollection>(
				 ps.getParameter<edm::InputTag>(
				   "calol2EGammaCollectionData"))),
    calol2EGammaCollectionEmul(consumes <l1t::EGammaBxCollection>(
				 ps.getParameter<edm::InputTag>(
				   "calol2EGammaCollectionEmul"))),
    calol2TauCollectionData(consumes <l1t::TauBxCollection>(
			      ps.getParameter<edm::InputTag>(
				"calol2TauCollectionData"))),
    calol2TauCollectionEmul(consumes <l1t::TauBxCollection>(
			      ps.getParameter<edm::InputTag>(
				"calol2TauCollectionEmul"))),
    calol2EtSumCollectionData(consumes <l1t::EtSumBxCollection>(
				ps.getParameter<edm::InputTag>(
				  "calol2EtSumCollectionData"))),
    calol2EtSumCollectionEmul(consumes <l1t::EtSumBxCollection>(
				ps.getParameter<edm::InputTag>(
				  "calol2EtSumCollectionEmul"))),
    calol2CaloTowerCollectionData(consumes <l1t::CaloTowerBxCollection>(
				    ps.getParameter<edm::InputTag>(
				      "calol2CaloTowerCollectionData"))),
    calol2CaloTowerCollectionEmul(consumes <l1t::CaloTowerBxCollection>(
				    ps.getParameter<edm::InputTag>(
				      "calol2CaloTowerCollectionEmul")))
{
  produces<l1t::JetBxCollection> ("dataJet").setBranchAlias("dataJets");
  produces<l1t::JetBxCollection> ("emulJet").setBranchAlias("emulJets");
  produces<l1t::EGammaBxCollection> ("dataEg").setBranchAlias("dataEgs");
  produces<l1t::EGammaBxCollection> ("emulEg").setBranchAlias("emulEgs");
  produces<l1t::TauBxCollection> ("dataTau").setBranchAlias("dataTaus");
  produces<l1t::TauBxCollection> ("emulTau").setBranchAlias("emulTaus");
  produces<l1t::EtSumBxCollection> ("dataEtSum").setBranchAlias("dataEtSums");
  produces<l1t::EtSumBxCollection> ("emulEtSum").setBranchAlias("emulEtSums");
  produces<l1t::CaloTowerBxCollection> ("dataCaloTower").setBranchAlias("dataCaloTowers");
  produces<l1t::CaloTowerBxCollection> ("emulCaloTower").setBranchAlias("emulCaloTowers");
}

void L1TStage2CaloLayer2Comp::produce (
  edm::Event& e,
  const edm::EventSetup & c) {

  bool eventGood = true;

  // map event contents to above collections
  e.getByToken(calol2JetCollectionData, jetDataCol);
  e.getByToken(calol2JetCollectionEmul, jetEmulCol);
  e.getByToken(calol2EGammaCollectionData, egDataCol);
  e.getByToken(calol2EGammaCollectionEmul, egEmulCol);
  e.getByToken(calol2TauCollectionData, tauDataCol);
  e.getByToken(calol2TauCollectionEmul, tauEmulCol);
  e.getByToken(calol2EtSumCollectionData, sumDataCol);
  e.getByToken(calol2EtSumCollectionEmul, sumEmulCol);
  e.getByToken(calol2CaloTowerCollectionData, caloTowerDataCol);
  e.getByToken(calol2CaloTowerCollectionEmul, caloTowerEmulCol);

  edm::LogProblem("l1tcalol2ebec") << "processing event " << e.id() << std::endl;

  // edm::LogProblem("l1tcalol2ebec") << "data jet col size: " << jetDataCol->size(currBx)
  // 	    << std::endl;
  // edm::LogProblem("l1tcalol2ebec") << "emul jet col size: " << jetEmulCol->size(currBx)
  // 	    << std::endl;

  // edm::LogProblem("l1tcalol2ebec") << "data eg col size: " << egDataCol->size(currBx)
  // 	    << std::endl;
  // edm::LogProblem("l1tcalol2ebec") << "emul eg col size: " << egEmulCol->size(currBx)
  // 	    << std::endl;

  // edm::LogProblem("l1tcalol2ebec") << "data tau col size: " << tauDataCol->size(currBx)
  // 	    << std::endl;
  // edm::LogProblem("l1tcalol2ebec") << "emul tau col size: " << tauEmulCol->size(currBx)
  // 	    << std::endl;

  // edm::LogProblem("l1tcalol2ebec") << "data sum col size: " << sumDataCol->size(currBx)
  // 	    << std::endl;
  // edm::LogProblem("l1tcalol2ebec") << "emul sum col size: " << sumEmulCol->size(currBx)
  // 	    << std::endl;

  using namespace l1t;

  // as a test store every event to edm file
  if (!compareJets(jetDataCol, jetEmulCol) || true) {
    eventGood = false;
    // TODO: Remove cout below once development is finished
  } else { edm::LogProblem("l1tcalol2ebec") << "jets good" << std::endl;}

  if (!compareEGs(egDataCol, egEmulCol) || true) {
    eventGood = false;
    // TODO: Remove cout below once development is finished
  } else { edm::LogProblem("l1tcalol2ebec") << "egs ok" << std::endl;}

  if (!compareTaus(tauDataCol, tauEmulCol) || true) {
    eventGood = false;
    // TODO: Remove cout below once development is finished
  } else { edm::LogProblem("l1tcalol2ebec") << "tau ok" << std::endl;}

  if (!compareSums(sumDataCol, sumEmulCol) || true) {
    eventGood = false;
    // TODO: Remove cout below once development is finished
  } else { edm::LogProblem("l1tcalol2ebec") << "sums ok" << std::endl;}

  if (!eventGood) {

    dumpEventToFile();

    // store all contents of event to edm file:
    dumpEventToEDM(e);
  } else {
    edm::LogProblem("l1tcalol2ebec") << " ====> Event good. " << std::endl;
  }
}

// comparison method for jets
bool L1TStage2CaloLayer2Comp::compareJets(
  const edm::Handle<l1t::JetBxCollection> & dataCol,
  const edm::Handle<l1t::JetBxCollection> & emulCol)
{
  bool eventGood = true;

  l1t::JetBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::JetBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  if (dataCol->size(currBx) > 0) {
    // sort data jets
    std::vector<l1t::Jet> jets;
    while (true) {

      jets.emplace_back(*dataIt);

      dataIt++;
      if (dataIt == dataCol->end(currBx))
	break;
    }

    accuSort(jets);
    dataIt = jets.begin();
  }

  // process jets
  if (dataCol->size(currBx) != emulCol->size(currBx)) {
    edm::LogProblem("l1tcalol2ebec")
      << "Jet collection size difference: "
      << "data = " << dataCol->size(currBx)
      << " emul = " << emulCol->size(currBx)
      << std::endl;
    return false;
  }

  int nJets = 0;
  if (dataIt != dataCol->end(currBx) ||
      emulIt != emulCol->end(currBx)) {
    while(true) {

      ++nJets;

      bool posGood = true;
      bool etGood = true;

      // object pt mismatch
      if (dataIt->hwPt() != emulIt->hwPt()) {
        etGood = false;
 	eventGood = false;
      }

      // object position mismatch (phi)
      if (dataIt->hwPhi() != emulIt->hwPhi()){
	posGood = false;
	eventGood = false;
      }

      // object position mismatch (eta)
      if (dataIt->hwEta() != emulIt->hwEta()) {
	posGood = false;
	eventGood = false;
      }

      // if both position and energy agree, jet is good
      if (etGood && posGood) {
	// agreementSummary->Fill(JETGOOD_S);
	// jetSummary->Fill(JETGOOD);
      } else {
	edm::LogProblem("l1tcalol2ebec")
	  << "Jet Problem (data emul): "
	  << "\tEt = " << dataIt->hwPt() << " " << emulIt->hwPt()
	  << "\teta = " << dataIt->hwEta() << " " << emulIt->hwEta()
	  << "\tphi = " << dataIt->hwPhi() << " " << emulIt->hwPhi()
	  << std::endl;
      }

      // increment position of pointers
      ++dataIt;
      ++emulIt;

      if (dataIt == dataCol->end(currBx) ||
	  emulIt == emulCol->end(currBx))
	break;
    }
  } else {
    if (dataCol->size(currBx) != 0 || emulCol->size(currBx) != 0)
      return false;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for e/gammas
bool L1TStage2CaloLayer2Comp::compareEGs(
  const edm::Handle<l1t::EGammaBxCollection> & dataCol,
  const edm::Handle<l1t::EGammaBxCollection> & emulCol)
{
  bool eventGood = true;

  l1t::EGammaBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::EGammaBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  if (dataCol->size(currBx) > 0) {
    // sort data egs
    std::vector<l1t::EGamma> egs;
    while (true) {

      egs.emplace_back(*dataIt);

      dataIt++;
      if (dataIt == dataCol->end(currBx))
	break;
    }
    accuSort(egs);
    dataIt = egs.begin();
  }

  // check length of collections
  if (dataCol->size(currBx) != emulCol->size(currBx)) {
    edm::LogProblem("l1tcalol2ebec")
      << "EG collection size difference: "
      << "data = " << dataCol->size(currBx)
      << " emul = " << emulCol->size(currBx)
      << std::endl;
    return false;
  }

  // processing continues only of length of data collections is the same
  if (dataIt != dataCol->end(currBx) ||
      emulIt != emulCol->end(currBx)) {

    while(true) {

      bool posGood = true;
      bool etGood = true;

      // object pt mismatch
      if (dataIt->hwPt() != emulIt->hwPt()) {
	etGood = false;
	eventGood = false;
      }

      // object position mismatch (phi)
      if (dataIt->hwPhi() != emulIt->hwPhi()) {
	posGood = false;
	eventGood = false;
      }

      // object position mismatch (eta)
      if (dataIt->hwEta() != emulIt->hwEta()) {
	posGood = false;
	eventGood = false;
      }

      // if both position and energy agree, object is good
      if (posGood && etGood) {
	// agreementSummary->Fill(EGGOOD_S);
      } else {
	edm::LogProblem("l1tcalol2ebec")
	  << "EG Problem (data emul): "
	  << "\tEt = " << dataIt->hwPt() << " " << emulIt->hwPt()
	  << "\teta = " << dataIt->hwEta() << " " << emulIt->hwEta()
	  << "\tphi = " << dataIt->hwPhi() << " " << emulIt->hwPhi()
	  << std::endl;
      }

      // increment position of pointers
      ++dataIt;
      ++emulIt;

      if (dataIt == dataCol->end(currBx) ||
	  emulIt == emulCol->end(currBx))
	break;
    }
  } else {
    if (dataCol->size(currBx) != 0 || emulCol->size(currBx) != 0)
      return false;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for taus
bool L1TStage2CaloLayer2Comp::compareTaus(
  const edm::Handle<l1t::TauBxCollection> & dataCol,
  const edm::Handle<l1t::TauBxCollection> & emulCol)
{
  bool eventGood = true;

  l1t::TauBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::TauBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  if (dataCol->size(currBx) > 0) {
    // sort data taus
    std::vector<l1t::Tau> taus;
    while (true) {

      taus.emplace_back(*dataIt);

      dataIt++;
      if (dataIt == dataCol->end(currBx))
	break;
    }
    accuSort(taus);
    dataIt = taus.begin();
  }

  // check length of collections
  if (dataCol->size(currBx) != emulCol->size(currBx)) {
    edm::LogProblem("l1tcalol2ebec")
      << "Tau collection size difference: "
      << "data = " << dataCol->size(currBx)
      << " emul = " << emulCol->size(currBx)
      << std::endl;
    return false;
  }

  // processing continues only of length of data collections is the same
  if (dataIt != dataCol->end(currBx) ||
      emulIt != emulCol->end(currBx)) {

    while(true) {

      bool posGood = true;
      bool etGood = true;

      // object Et mismatch
      if (dataIt->hwPt() != emulIt->hwPt()) {
	etGood = false;
	eventGood = false;
      }

      // object position mismatch (phi)
      if (dataIt->hwPhi() != emulIt->hwPhi()) {
	posGood = false;
	eventGood = false;
      }

      // object position mismatch (eta)
      if (dataIt->hwEta() != emulIt->hwEta()) {
	posGood = false;
	eventGood = false;
      }

      // if both position and energy agree, object is good
      if (posGood && etGood) {
	// agreementSummary->Fill(TAUGOOD_S);
      } else {
	edm::LogProblem("l1tcalol2ebec")
	  << "Tau Problem (data emul): "
	  << "\tEt = " << dataIt->hwPt() << " " << emulIt->hwPt()
	  << "\teta = " << dataIt->hwEta() << " " << emulIt->hwEta()
	  << "\tphi = " << dataIt->hwPhi() << " " << emulIt->hwPhi()
	  << std::endl;
      }
      // increment position of pointers
      ++dataIt;
      ++emulIt;

      if (dataIt == dataCol->end(currBx) ||
	  emulIt == emulCol->end(currBx))
	break;
    }
  } else {
    if (dataCol->size(currBx) != 0 || emulCol->size(currBx) != 0)
      return false;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for sums
bool L1TStage2CaloLayer2Comp::compareSums(
  const edm::Handle<l1t::EtSumBxCollection> & dataCol,
  const edm::Handle<l1t::EtSumBxCollection> & emulCol)
{
  bool eventGood = true;

  double dataEt = 0;
  double emulEt = 0;

  l1t::EtSumBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::EtSumBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  // if either data or emulator collections are empty, or they have different
  // size, mark the event as bad (this should never occur in normal running)
  if ((dataCol->isEmpty(currBx)) || (emulCol->isEmpty(currBx)) ||
      (dataCol->size(currBx) != emulCol->size(currBx))) {
    edm::LogProblem("l1tcalol2ebec")
      << "EtSum collection size difference: "
      << "data = " << dataCol->size(currBx)
      << " emul = " << emulCol->size(currBx)
      << std::endl;

    return false;
  }

  l1t::EtSum dataSum;
  l1t::EtSum emulSum;
  for (unsigned int i = 0; i < emulCol->size(currBx); i++) {

    emulSum = emulCol->at(currBx, i);
    dataSum = dataCol->at(currBx, emul_to_data_sum_index_map[i]);

    if (emulSum.getType() != dataSum.getType()) {
      edm::LogProblem("l1tcalol2ebec")
	<< "EtSum type problem (data emul): "
	<< dataSum.getType() << " " << emulSum.getType()
	<< std::endl;
    }


    dataEt = dataSum.hwPt();
    emulEt = emulSum.hwPt();

    if (dataEt != emulEt) {
      eventGood = false;
    }

    switch (emulSum.getType()) {
    case l1t::EtSum::EtSumType::kTotalEt:     // ETT (enum val = 0)
      if (!eventGood) {
	edm::LogProblem("l1tcalol2ebec") << "ETT sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalEtx:    // ETTx (enum val = 4)
      if (!eventGood) {
	edm::LogProblem("l1tcalol2ebec") << "ETTx sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalEty:    // ETTy (enum val = 5)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "ETTy sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalEtHF:   // ETTHF (enum val = 15)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "ETTHF sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalEtxHF:  // ETTHFx (enum val = 9)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "ETTHFx sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalEtyHF:  // ETTHFy (enum val = 10)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "ETTHFy sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalEtEm:   // ETTEM (enum val = 16)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "ETTEM sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kMinBiasHFP0: // MBHFP0 (enum val = 11)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "MBHFP0 sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kMinBiasHFP1: // MBHFP1 (enum val = 13)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "MBHFP1 sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kMinBiasHFM0: // MBHFM0 (enum val = 12)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "MBHFM0 sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kMinBiasHFM1: // MBHFM1 (enum val = 14)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "MBHFM1 sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTowerCount:  // TowerCount (enum val = 21)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "TowerCount sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalHt:     // HTT (enum val = 1)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "HTT sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalHtx:    // HTTx (enum val = 6)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "HTTx sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalHty:    // HTTy (enum val = 7)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "HTTy sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalHtHF:   // HTTHF (enum val = 17)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "HTTHF sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalHtxHF:  // HTTHFx (enum val = 18)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "HTTHFx sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kTotalHtyHF:  // HTTHFy (enum val = 19)
      if (!eventGood) {
	eventGood = false;
	edm::LogProblem("l1tcalol2ebec") << "HTTHFy sum bad" << std::endl;
      }
      break;
    case l1t::EtSum::EtSumType::kMissingEt:   // MET (enum val = 9)
    case l1t::EtSum::EtSumType::kMissingHt:   // MHT (enum val = 11)
    case l1t::EtSum::EtSumType::kMissingEtHF: // METHF (enum val = 10)
    case l1t::EtSum::EtSumType::kMissingHtHF: // MHTHF (enum val = 12)
      // this module currently compares only MP outputs, the above 4 sume types
      // are produced by the Demux board in the CaloLayer2 system and are
      // included here only for completeness (so that switch can be used with an
      // enum)
      break;
    }

    // edm::LogProblem("l1tcalol2ebec")
    //   << "Checking sums, sum type = " << emulCol->at(currBx, i).getType()
    //   << " " << dataCol->at(currBx, emul_to_data_sum_index_map[i]).getType()
    //   << std::endl;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// sort for jets
void L1TStage2CaloLayer2Comp::accuSort(std::vector<l1t::Jet> & jets){

  math::PtEtaPhiMLorentzVector emptyP4;
  l1t::Jet tempJet (emptyP4, 0, 0, 0, 0);
  std::vector< std::vector<l1t::Jet> > jetEtaPos( 41 , std::vector<l1t::Jet>(18, tempJet));
  std::vector< std::vector<l1t::Jet> > jetEtaNeg( 41 , std::vector<l1t::Jet>(18, tempJet));

  for (unsigned int iJet = 0; iJet < jets.size(); iJet++) {
    if (jets.at(iJet).hwEta() > 0) jetEtaPos.at(jets.at(iJet).hwEta()-1).at((jets.at(iJet).hwPhi()-1)/4) = jets.at(iJet);
    else  jetEtaNeg.at(-(jets.at(iJet).hwEta()+1)).at((jets.at(iJet).hwPhi()-1)/4) = jets.at(iJet);
  }

  AccumulatingSort <l1t::Jet> etaPosSorter(7);
  AccumulatingSort <l1t::Jet> etaNegSorter(7);
  std::vector<l1t::Jet> accumEtaPos;
  std::vector<l1t::Jet> accumEtaNeg;

  for( int ieta = 0 ; ieta < 41 ; ++ieta) {
    // eta +
    std::vector<l1t::Jet>::iterator start_, end_;
    start_ = jetEtaPos.at(ieta).begin();
    end_   = jetEtaPos.at(ieta).end();
    BitonicSort<l1t::Jet>(down, start_, end_);
    etaPosSorter.Merge( jetEtaPos.at(ieta) , accumEtaPos );

    // eta -
    start_ = jetEtaNeg.at(ieta).begin();
    end_   = jetEtaNeg.at(ieta).end();
    BitonicSort<l1t::Jet>(down, start_, end_);
    etaNegSorter.Merge( jetEtaNeg.at(ieta) , accumEtaNeg );

  }

  //check for 6 & 7th jets with same et and eta. Keep jet with larger phi
  if(accumEtaPos.at(6).hwPt()==accumEtaPos.at(5).hwPt() && accumEtaPos.at(6).hwEta()==accumEtaPos.at(5).hwEta()
     && accumEtaPos.at(6).hwPhi() > accumEtaPos.at(5).hwPhi()){
    accumEtaPos.at(5)=accumEtaPos.at(6);
  }
  if(accumEtaNeg.at(6).hwPt()==accumEtaNeg.at(5).hwPt() && accumEtaNeg.at(6).hwEta()==accumEtaNeg.at(5).hwEta()
     && accumEtaNeg.at(6).hwPhi() > accumEtaNeg.at(5).hwPhi()){
    accumEtaNeg.at(5)=accumEtaNeg.at(6);
  }

  //truncate
  accumEtaPos.resize(6);
  accumEtaNeg.resize(6);

  // put all 12 candidates in the original jet vector, removing zero energy ones
  jets.clear();
  for (l1t::Jet accjet : accumEtaPos) {
    if (accjet.hwPt() > 0) jets.push_back(accjet);
  }
  for (l1t::Jet accjet : accumEtaNeg) {
    if (accjet.hwPt() > 0) jets.push_back(accjet);
  }
}

// sort for eg
void L1TStage2CaloLayer2Comp::accuSort(std::vector<l1t::EGamma> & jets){

  math::PtEtaPhiMLorentzVector emptyP4;
  l1t::EGamma tempEGamma (emptyP4, 0, 0, 0, 0);
  std::vector< std::vector<l1t::EGamma> > jetEtaPos( 41 , std::vector<l1t::EGamma>(18, tempEGamma));
  std::vector< std::vector<l1t::EGamma> > jetEtaNeg( 41 , std::vector<l1t::EGamma>(18, tempEGamma));

  for (unsigned int iEGamma = 0; iEGamma < jets.size(); iEGamma++) {
    if (jets.at(iEGamma).hwEta() > 0) jetEtaPos.at(jets.at(iEGamma).hwEta()-1).at((jets.at(iEGamma).hwPhi()-1)/4) = jets.at(iEGamma);
    else  jetEtaNeg.at(-(jets.at(iEGamma).hwEta()+1)).at((jets.at(iEGamma).hwPhi()-1)/4) = jets.at(iEGamma);
  }

  AccumulatingSort <l1t::EGamma> etaPosSorter(7);
  AccumulatingSort <l1t::EGamma> etaNegSorter(7);
  std::vector<l1t::EGamma> accumEtaPos;
  std::vector<l1t::EGamma> accumEtaNeg;

  for( int ieta = 0 ; ieta < 41 ; ++ieta) {
    // eta +
    std::vector<l1t::EGamma>::iterator start_, end_;
    start_ = jetEtaPos.at(ieta).begin();
    end_   = jetEtaPos.at(ieta).end();
    BitonicSort<l1t::EGamma>(down, start_, end_);
    etaPosSorter.Merge( jetEtaPos.at(ieta) , accumEtaPos );

    // eta -
    start_ = jetEtaNeg.at(ieta).begin();
    end_   = jetEtaNeg.at(ieta).end();
    BitonicSort<l1t::EGamma>(down, start_, end_);
    etaNegSorter.Merge( jetEtaNeg.at(ieta) , accumEtaNeg );

  }

  //check for 6 & 7th jets with same et and eta. Keep jet with larger phi
  if(accumEtaPos.at(6).hwPt()==accumEtaPos.at(5).hwPt() && accumEtaPos.at(6).hwEta()==accumEtaPos.at(5).hwEta()
     && accumEtaPos.at(6).hwPhi() > accumEtaPos.at(5).hwPhi()){
    accumEtaPos.at(5)=accumEtaPos.at(6);
  }
  if(accumEtaNeg.at(6).hwPt()==accumEtaNeg.at(5).hwPt() && accumEtaNeg.at(6).hwEta()==accumEtaNeg.at(5).hwEta()
     && accumEtaNeg.at(6).hwPhi() > accumEtaNeg.at(5).hwPhi()){
    accumEtaNeg.at(5)=accumEtaNeg.at(6);
  }

  //truncate
  accumEtaPos.resize(6);
  accumEtaNeg.resize(6);

  // put all 12 candidates in the original jet vector, removing zero energy ones
  jets.clear();
  for (l1t::EGamma accjet : accumEtaPos) {
    if (accjet.hwPt() > 0) jets.push_back(accjet);
  }
  for (l1t::EGamma accjet : accumEtaNeg) {
    if (accjet.hwPt() > 0) jets.push_back(accjet);
  }
}


// sort for tau
void L1TStage2CaloLayer2Comp::accuSort(std::vector<l1t::Tau> & jets){

  math::PtEtaPhiMLorentzVector emptyP4;
  l1t::Tau tempTau (emptyP4, 0, 0, 0, 0);
  std::vector< std::vector<l1t::Tau> > jetEtaPos( 41 , std::vector<l1t::Tau>(18, tempTau));
  std::vector< std::vector<l1t::Tau> > jetEtaNeg( 41 , std::vector<l1t::Tau>(18, tempTau));

  for (unsigned int iTau = 0; iTau < jets.size(); iTau++) {
    if (jets.at(iTau).hwEta() > 0) jetEtaPos.at(jets.at(iTau).hwEta()-1).at((jets.at(iTau).hwPhi()-1)/4) = jets.at(iTau);
    else  jetEtaNeg.at(-(jets.at(iTau).hwEta()+1)).at((jets.at(iTau).hwPhi()-1)/4) = jets.at(iTau);
  }

  AccumulatingSort <l1t::Tau> etaPosSorter(7);
  AccumulatingSort <l1t::Tau> etaNegSorter(7);
  std::vector<l1t::Tau> accumEtaPos;
  std::vector<l1t::Tau> accumEtaNeg;

  for( int ieta = 0 ; ieta < 41 ; ++ieta) {
    // eta +
    std::vector<l1t::Tau>::iterator start_, end_;
    start_ = jetEtaPos.at(ieta).begin();
    end_   = jetEtaPos.at(ieta).end();
    BitonicSort<l1t::Tau>(down, start_, end_);
    etaPosSorter.Merge( jetEtaPos.at(ieta) , accumEtaPos );

    // eta -
    start_ = jetEtaNeg.at(ieta).begin();
    end_   = jetEtaNeg.at(ieta).end();
    BitonicSort<l1t::Tau>(down, start_, end_);
    etaNegSorter.Merge( jetEtaNeg.at(ieta) , accumEtaNeg );

  }

  //check for 6 & 7th jets with same et and eta. Keep jet with larger phi
  if(accumEtaPos.at(6).hwPt()==accumEtaPos.at(5).hwPt() && accumEtaPos.at(6).hwEta()==accumEtaPos.at(5).hwEta()
     && accumEtaPos.at(6).hwPhi() > accumEtaPos.at(5).hwPhi()){
    accumEtaPos.at(5)=accumEtaPos.at(6);
  }
  if(accumEtaNeg.at(6).hwPt()==accumEtaNeg.at(5).hwPt() && accumEtaNeg.at(6).hwEta()==accumEtaNeg.at(5).hwEta()
     && accumEtaNeg.at(6).hwPhi() > accumEtaNeg.at(5).hwPhi()){
    accumEtaNeg.at(5)=accumEtaNeg.at(6);
  }

  //truncate
  accumEtaPos.resize(6);
  accumEtaNeg.resize(6);

  // put all 12 candidates in the original jet vector, removing zero energy ones
  jets.clear();
  for (l1t::Tau accjet : accumEtaPos) {
    if (accjet.hwPt() > 0) jets.push_back(accjet);
  }
  for (l1t::Tau accjet : accumEtaNeg) {
    if (accjet.hwPt() > 0) jets.push_back(accjet);
  }
}

void L1TStage2CaloLayer2Comp::dumpEventToFile() {
  
}


void L1TStage2CaloLayer2Comp::dumpEventToEDM(edm::Event &e) {



    // store all jets to an edm file
  std::unique_ptr<l1t::JetBxCollection> mpjets_data (new l1t::JetBxCollection(0, currBx, currBx));
  std::unique_ptr<l1t::JetBxCollection> mpjets_emul (new l1t::JetBxCollection(0, currBx, currBx));

    // std::unique_ptr<std::vector<Jet>> localJetsData(new std::vector<Jet>);
    // std::unique_ptr<std::vector<Jet>> localJetsEmul(new std::vector<Jet>);

    for (auto jet = jetDataCol->begin(currBx); jet != jetDataCol->end(currBx); ++jet)
      mpjets_data->push_back(0, (*jet));
    for (auto jet = jetEmulCol->begin(currBx); jet != jetEmulCol->end(currBx); ++jet)
      mpjets_emul->push_back(0, (*jet));

    e.put(std::move(mpjets_data), "dataJet");
    e.put(std::move(mpjets_emul), "emulJet");

    // store all egs to an edm file
    std::unique_ptr<l1t::EGammaBxCollection> mpEGammas_data (new l1t::EGammaBxCollection(0, currBx, currBx));
    std::unique_ptr<l1t::EGammaBxCollection> mpEGammas_emul (new l1t::EGammaBxCollection(0, currBx, currBx));

    // std::unique_ptr<std::vector<EGamma>> localEGammasData(new std::vector<EGamma>);
    // std::unique_ptr<std::vector<EGamma>> localEGammasEmul(new std::vector<EGamma>);

    for (auto eg = egDataCol->begin(currBx); eg != egDataCol->end(currBx); ++eg)
      mpEGammas_data->push_back(0, (*eg));
    for (auto eg = egEmulCol->begin(currBx); eg != egEmulCol->end(currBx); ++eg)
      mpEGammas_emul->push_back(0, (*eg));

    e.put(std::move(mpEGammas_data), "dataEg");
    e.put(std::move(mpEGammas_emul), "emulEg");

    // store all taus to an edm file
    std::unique_ptr<l1t::TauBxCollection> mptaus_data (new l1t::TauBxCollection(0, currBx, currBx));
    std::unique_ptr<l1t::TauBxCollection> mptaus_emul (new l1t::TauBxCollection(0, currBx, currBx));

    // std::unique_ptr<std::vector<Tau>> localTausData(new std::vector<Tau>);
    // std::unique_ptr<std::vector<Tau>> localTausEmul(new std::vector<Tau>);

    for (auto tau = tauDataCol->begin(currBx); tau != tauDataCol->end(currBx); ++tau)
      mptaus_data->push_back(0, (*tau));
    for (auto tau = tauEmulCol->begin(currBx); tau != tauEmulCol->end(currBx); ++tau)
      mptaus_emul->push_back(0, (*tau));

    e.put(std::move(mptaus_data), "dataTau");
    e.put(std::move(mptaus_emul), "emulTau");

    // store all sums to an edm file
    std::unique_ptr<l1t::EtSumBxCollection> mpsums_data (new l1t::EtSumBxCollection(0, currBx, currBx));
    std::unique_ptr<l1t::EtSumBxCollection> mpsums_emul (new l1t::EtSumBxCollection(0, currBx, currBx));

    // std::unique_ptr<std::vector<EtSum>> localSumsData(new std::vector<EtSum>);
    // std::unique_ptr<std::vector<EtSum>> localSumsEmul(new std::vector<EtSum>);

    for (auto sum = sumDataCol->begin(currBx); sum != sumDataCol->end(currBx); ++sum)
      mpsums_data->push_back(0, (*sum));
    for (auto sum = sumEmulCol->begin(currBx); sum != sumEmulCol->end(currBx); ++sum)
      mpsums_emul->push_back(0, (*sum));

    e.put(std::move(mpsums_data), "dataEtSum");
    e.put(std::move(mpsums_emul), "emulEtSum");

    // store calorimeter towers
    std::unique_ptr<l1t::CaloTowerBxCollection> mptowers_data (new l1t::CaloTowerBxCollection(0, currBx, currBx));
    std::unique_ptr<l1t::CaloTowerBxCollection> mptowers_emul (new l1t::CaloTowerBxCollection(0, currBx, currBx));

    // std::unique_ptr<std::vector<CaloTower>> localTowersData(new std::vector<CaloTower>);
    // std::unique_ptr<std::vector<CaloTower>> localTowersEmul(new std::vector<CaloTower>);

    for (auto tower = caloTowerDataCol->begin(currBx); tower != caloTowerDataCol->end(currBx); ++tower)
      mptowers_data->push_back(0, (*tower));
    for (auto tower = caloTowerEmulCol->begin(currBx); tower != caloTowerEmulCol->end(currBx); ++tower)
      mptowers_emul->push_back(0, (*tower));

    e.put(std::move(mptowers_data), "dataCaloTower");
    e.put(std::move(mptowers_emul), "emulCaloTower");
}

DEFINE_FWK_MODULE (L1TStage2CaloLayer2Comp);

#endif
