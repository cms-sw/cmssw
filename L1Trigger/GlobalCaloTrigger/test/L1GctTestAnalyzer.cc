#include "L1Trigger/GlobalCaloTrigger/test/L1GctTestAnalyzer.h"

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

using std::endl;
using std::ios;
using std::string;

//
// constructors and destructor
//
L1GctTestAnalyzer::L1GctTestAnalyzer(const edm::ParameterSet& iConfig)
    : rawLabel_(iConfig.getUntrackedParameter<edm::InputTag>("rawInput", edm::InputTag("L1GctRawDigis"))),
      emuLabel_(iConfig.getUntrackedParameter<edm::InputTag>("emuInput", edm::InputTag("L1GctEmuDigis"))),
      outFilename_(iConfig.getUntrackedParameter<string>("outFile", "gctAnalyzer.txt")),
      doHW_(iConfig.getUntrackedParameter<bool>("doHardware", true)),
      doEmu_(iConfig.getUntrackedParameter<bool>("doEmulated", true)),
      doRctEM_(iConfig.getUntrackedParameter<bool>("doRctEm", true)),
      doInternEM_(iConfig.getUntrackedParameter<bool>("doInternEm", true)),
      doEM_(iConfig.getUntrackedParameter<bool>("doEm", true)),
      doJets_(iConfig.getUntrackedParameter<bool>("doJets", false)),
      doEnergySums_(iConfig.getUntrackedParameter<bool>("doEnergySums", false)),
      rctEmMinRank_(iConfig.getUntrackedParameter<unsigned>("rctEmMinRank", 0)) {
  //now do what ever initialization is needed

  outFile_.open(outFilename_.c_str(), ios::out);
}

L1GctTestAnalyzer::~L1GctTestAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  outFile_.close();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1GctTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  outFile_ << "Run :" << iEvent.id().run() << "  Event :" << iEvent.id().event() << endl;

  if (doRctEM_ && doHW_) {
    doRctEM(iEvent, rawLabel_);
  }
  if (doRctEM_ && doEmu_) {
    doRctEM(iEvent, emuLabel_);
  }
  if (doInternEM_ && doHW_) {
    doInternEM(iEvent, rawLabel_);
  }
  if (doInternEM_ && doEmu_) {
  }  //doInternEM(iEvent, emuLabel_); }
  if (doEM_ && doHW_) {
    doEM(iEvent, rawLabel_);
  }
  if (doEM_ && doEmu_) {
    doEM(iEvent, emuLabel_);
  }
  if (doJets_ && doHW_) {
    doJets(iEvent, rawLabel_);
  }
  if (doJets_ && doEmu_) {
    doJets(iEvent, emuLabel_);
  }
  if (doEnergySums_ && doHW_) {
    doEnergySums(iEvent, rawLabel_);
  }
  if (doEnergySums_ && doEmu_) {
    doEnergySums(iEvent, emuLabel_);
  }
}

void L1GctTestAnalyzer::doEM(const edm::Event& iEvent, edm::InputTag label) {
  using namespace edm;

  Handle<L1GctEmCandCollection> isoEm;
  Handle<L1GctEmCandCollection> nonIsoEm;

  L1GctEmCandCollection::const_iterator ie;
  L1GctEmCandCollection::const_iterator ne;

  iEvent.getByLabel(label.label(), "isoEm", isoEm);
  iEvent.getByLabel(label.label(), "nonIsoEm", nonIsoEm);

  outFile_ << "Iso EM :"
           << " from : " << label.label() << endl;
  for (ie = isoEm->begin(); ie != isoEm->end(); ie++) {
    outFile_ << (*ie) << endl;
  }
  outFile_ << endl;

  outFile_ << "Non-iso EM :"
           << " from : " << label.label() << endl;
  for (ne = nonIsoEm->begin(); ne != nonIsoEm->end(); ne++) {
    outFile_ << (*ne) << endl;
  }
  outFile_ << endl;
}

void L1GctTestAnalyzer::doRctEM(const edm::Event& iEvent, edm::InputTag label) {
  using namespace edm;

  Handle<L1CaloEmCollection> em;

  L1CaloEmCollection::const_iterator e;

  iEvent.getByLabel(label.label(), "", em);

  outFile_ << "RCT EM :"
           << " from : " << label.label() << endl;
  for (e = em->begin(); e != em->end(); e++) {
    if (e->rank() >= rctEmMinRank_) {
      outFile_ << (*e) << endl;
    }
  }
  outFile_ << endl;
}

void L1GctTestAnalyzer::doInternEM(const edm::Event& iEvent, edm::InputTag label) {
  using namespace edm;

  Handle<L1GctInternEmCandCollection> em;

  L1GctInternEmCandCollection::const_iterator e;

  iEvent.getByLabel(label.label(), "", em);

  outFile_ << "Internal EM :"
           << " from : " << label.label() << endl;
  for (e = em->begin(); e != em->end(); e++) {
    outFile_ << (*e) << endl;
  }
  outFile_ << endl;
}

void L1GctTestAnalyzer::doJets(const edm::Event& iEvent, edm::InputTag label) {
  using namespace edm;

  Handle<L1GctJetCandCollection> cenJets;
  Handle<L1GctJetCandCollection> forJets;
  Handle<L1GctJetCandCollection> tauJets;

  L1GctJetCandCollection::const_iterator cj;
  L1GctJetCandCollection::const_iterator fj;
  L1GctJetCandCollection::const_iterator tj;

  iEvent.getByLabel(label.label(), "cenJets", cenJets);
  iEvent.getByLabel(label.label(), "forJets", forJets);
  iEvent.getByLabel(label.label(), "tauJets", tauJets);

  outFile_ << "Central jets :"
           << " from : " << label.label() << endl;
  for (cj = cenJets->begin(); cj != cenJets->end(); cj++) {
    outFile_ << (*cj) << endl;
  }
  outFile_ << endl;

  outFile_ << "Forward jets : "
           << " from : " << label.label() << endl;
  for (fj = forJets->begin(); fj != forJets->end(); fj++) {
    outFile_ << (*fj) << endl;
  }
  outFile_ << endl;

  outFile_ << "Tau jets :"
           << " from : " << label.label() << endl;
  for (tj = tauJets->begin(); tj != tauJets->end(); tj++) {
    outFile_ << (*tj) << endl;
  }
  outFile_ << endl;
}

void L1GctTestAnalyzer::doEnergySums(const edm::Event& iEvent, edm::InputTag label) {
  using namespace edm;

  Handle<L1GctEtTotal> etTotResult;
  Handle<L1GctEtHad> etHadResult;
  Handle<L1GctEtMiss> etMissResult;

  iEvent.getByLabel(label, etTotResult);
  iEvent.getByLabel(label, etHadResult);
  iEvent.getByLabel(label, etMissResult);

  outFile_ << "Total Et from : " << label.label() << endl;
  outFile_ << (*etTotResult) << endl;
  outFile_ << "Total Ht from : " << label.label() << endl;
  outFile_ << (*etHadResult) << endl;
  outFile_ << "Missing Et from : " << label.label() << endl;
  outFile_ << (*etMissResult) << endl;

  outFile_ << endl;
}
