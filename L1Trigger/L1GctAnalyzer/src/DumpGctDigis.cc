#include "L1Trigger/L1GctAnalyzer/interface/DumpGctDigis.h"

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

using std::endl;
using std::ios;
using std::string;

//
// constructors and destructor
//
DumpGctDigis::DumpGctDigis(const edm::ParameterSet& iConfig)
    : rawLabel_(iConfig.getUntrackedParameter<edm::InputTag>("rawInput", edm::InputTag("L1GctRawDigis"))),
      emuRctLabel_(iConfig.getUntrackedParameter<edm::InputTag>("emuRctInput", edm::InputTag("L1RctEmuDigis"))),
      emuGctLabel_(iConfig.getUntrackedParameter<edm::InputTag>("emuGctInput", edm::InputTag("L1GctEmuDigis"))),
      outFilename_(iConfig.getUntrackedParameter<string>("outFile", "gctAnalyzer.txt")),
      doHW_(iConfig.getUntrackedParameter<bool>("doHardware", true)),
      doEmu_(iConfig.getUntrackedParameter<bool>("doEmulated", true)),
      doRctEM_(iConfig.getUntrackedParameter<bool>("doRctEm", true)),
      doEM_(iConfig.getUntrackedParameter<bool>("doEm", true)),
      doRegions_(iConfig.getUntrackedParameter<bool>("doRegions", false)),
      doJets_(iConfig.getUntrackedParameter<bool>("doJets", false)),
      doInternEM_(iConfig.getUntrackedParameter<bool>("doInternEm", true)),
      doFibres_(iConfig.getUntrackedParameter<bool>("doFibres", false)),
      doEnergySums_(iConfig.getUntrackedParameter<bool>("doEnergySums", false)),
      rctEmMinRank_(iConfig.getUntrackedParameter<unsigned>("rctEmMinRank", 0)) {
  //now do what ever initialization is needed

  outFile_.open(outFilename_.c_str(), ios::out);
}

DumpGctDigis::~DumpGctDigis() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  outFile_.close();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void DumpGctDigis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  outFile_ << "Run :" << iEvent.id().run() << "  Event :" << iEvent.id().event() << endl;

  // EM
  if (doRctEM_ && doHW_) {
    doRctEM(iEvent, rawLabel_);
  }
  if (doRctEM_ && doEmu_) {
    doRctEM(iEvent, emuRctLabel_);
  }
  if (doEM_ && doHW_) {
    doEM(iEvent, rawLabel_);
  }
  if (doEM_ && doEmu_) {
    doEM(iEvent, emuGctLabel_);
  }

  // Jets
  if (doRegions_ && doHW_) {
    doRegions(iEvent, rawLabel_);
  }
  if (doRegions_ && doEmu_) {
    doRegions(iEvent, emuRctLabel_);
  }
  if (doJets_ && doHW_) {
    doJets(iEvent, rawLabel_);
  }
  if (doJets_ && doEmu_) {
    doJets(iEvent, emuGctLabel_);
  }

  // Energy Sums
  if (doEnergySums_ && doHW_) {
    doEnergySums(iEvent, rawLabel_);
  }
  if (doEnergySums_ && doEmu_) {
    doEnergySums(iEvent, emuGctLabel_);
  }

  // debugging
  if (doInternEM_ && doHW_) {
    doInternEM(iEvent, rawLabel_);
  }
  if (doFibres_ && doHW_) {
    doFibres(iEvent, rawLabel_);
  }
}

void DumpGctDigis::doEM(const edm::Event& iEvent, const edm::InputTag& label) {
  using namespace edm;

  Handle<L1GctEmCandCollection> isoEm;
  Handle<L1GctEmCandCollection> nonIsoEm;

  L1GctEmCandCollection::const_iterator ie;
  L1GctEmCandCollection::const_iterator ne;

  iEvent.getByLabel(label.label(), "isoEm", isoEm);
  iEvent.getByLabel(label.label(), "nonIsoEm", nonIsoEm);

  outFile_ << "Iso EM from : " << label.label() << endl;
  for (ie = isoEm->begin(); ie != isoEm->end(); ie++) {
    outFile_ << (*ie) << " ieta(detID)=" << ie->regionId().ieta() << " iphi(detID)=" << ie->regionId().iphi() << endl;
  }
  outFile_ << endl;

  outFile_ << "Non-iso EM from : " << label.label() << endl;
  for (ne = nonIsoEm->begin(); ne != nonIsoEm->end(); ne++) {
    outFile_ << (*ne) << " ieta(detID)=" << ne->regionId().ieta() << " iphi(detID)=" << ne->regionId().iphi() << endl;
  }
  outFile_ << endl;
}

void DumpGctDigis::doRctEM(const edm::Event& iEvent, const edm::InputTag& label) {
  using namespace edm;

  Handle<L1CaloEmCollection> em;

  L1CaloEmCollection::const_iterator e;

  iEvent.getByLabel(label, em);

  outFile_ << "RCT EM from : " << label.label() << endl;
  for (e = em->begin(); e != em->end(); e++) {
    if (e->rank() >= rctEmMinRank_) {
      outFile_ << (*e) << " ieta(detID)=" << e->regionId().ieta() << " iphi(detID)=" << e->regionId().iphi() << endl;
    }
  }
  outFile_ << endl;
}

void DumpGctDigis::doRegions(const edm::Event& iEvent, const edm::InputTag& label) {
  using namespace edm;

  Handle<L1CaloRegionCollection> rgns;

  L1CaloRegionCollection::const_iterator r;

  iEvent.getByLabel(label, rgns);

  outFile_ << "Regions from : " << label.label() << endl;
  for (r = rgns->begin(); r != rgns->end(); r++) {
    outFile_ << (*r) << endl;
  }
  outFile_ << endl;
}

void DumpGctDigis::doJets(const edm::Event& iEvent, const edm::InputTag& label) {
  using namespace edm;

  Handle<L1GctJetCandCollection> cenJets;
  Handle<L1GctJetCandCollection> forJets;
  Handle<L1GctJetCandCollection> tauJets;

  L1GctJetCandCollection::const_iterator cj;
  L1GctJetCandCollection::const_iterator fj;
  L1GctJetCandCollection::const_iterator tj;

  const std::string& labelStr = label.label();

  iEvent.getByLabel(labelStr, "cenJets", cenJets);
  iEvent.getByLabel(labelStr, "forJets", forJets);
  iEvent.getByLabel(labelStr, "tauJets", tauJets);

  outFile_ << "Central jets from : " << labelStr << endl;
  for (cj = cenJets->begin(); cj != cenJets->end(); cj++) {
    outFile_ << (*cj) << endl;
  }
  outFile_ << endl;

  outFile_ << "Forward jets from : " << labelStr << endl;
  for (fj = forJets->begin(); fj != forJets->end(); fj++) {
    outFile_ << (*fj) << endl;
  }
  outFile_ << endl;

  outFile_ << "Tau jets from : " << labelStr << endl;
  for (tj = tauJets->begin(); tj != tauJets->end(); tj++) {
    outFile_ << (*tj) << endl;
  }
}

void DumpGctDigis::doInternEM(const edm::Event& iEvent, const edm::InputTag& label) {
  using namespace edm;

  Handle<L1GctInternEmCandCollection> em;

  L1GctInternEmCandCollection::const_iterator e;

  iEvent.getByLabel(label, em);

  outFile_ << "Internal EM from : " << label.label() << endl;
  for (e = em->begin(); e != em->end(); e++) {
    outFile_ << (*e) << " ieta(detID)=" << e->regionId().ieta() << " iphi(detID)=" << e->regionId().iphi() << endl;
  }
  outFile_ << endl;
}

void DumpGctDigis::doFibres(const edm::Event& iEvent, const edm::InputTag& label) {
  using namespace edm;

  Handle<L1GctFibreCollection> fibres;

  L1GctFibreCollection::const_iterator f;

  iEvent.getByLabel(label, fibres);

  outFile_ << "Fibres from : " << label.label() << endl;
  for (f = fibres->begin(); f != fibres->end(); f++) {
    outFile_ << (*f) << endl;
  }
  outFile_ << endl;
}

void DumpGctDigis::doEnergySums(const edm::Event& iEvent, const edm::InputTag& label) {
  using namespace edm;

  Handle<L1GctEtTotalCollection> etTotal;
  Handle<L1GctEtHadCollection> etHad;
  Handle<L1GctEtMissCollection> etMiss;
  Handle<L1GctHtMissCollection> htMiss;

  iEvent.getByLabel(label, etTotal);
  iEvent.getByLabel(label, etHad);
  iEvent.getByLabel(label, etMiss);
  iEvent.getByLabel(label, htMiss);

  outFile_ << "Energy sums from: " << label.label() << endl;

  L1GctEtTotalCollection::const_iterator et;
  for (et = etTotal->begin(); et != etTotal->end(); et++) {
    outFile_ << *(et) << endl;
  }

  L1GctEtHadCollection::const_iterator ht;
  for (ht = etHad->begin(); ht != etHad->end(); ht++) {
    outFile_ << *(ht) << endl;
  }

  L1GctEtMissCollection::const_iterator met;
  for (met = etMiss->begin(); met != etMiss->end(); met++) {
    outFile_ << *(met) << endl;
  }

  L1GctHtMissCollection::const_iterator mht;
  for (mht = htMiss->begin(); mht != htMiss->end(); mht++) {
    outFile_ << *(mht) << endl;
  }
}
