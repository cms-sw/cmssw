#include "L1Trigger/L1GctAnalyzer/interface/DumpGctDigis.h"

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
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

using std::string;
using std::ios;
using std::endl;

//
// constructors and destructor
//
DumpGctDigis::DumpGctDigis(const edm::ParameterSet& iConfig) :
  rawLabel_( iConfig.getUntrackedParameter<edm::InputTag>("rawInput", edm::InputTag("L1GctRawDigis") ) ),
  emuRctLabel_( iConfig.getUntrackedParameter<edm::InputTag>("emuRctInput", edm::InputTag("L1RctEmuDigis") ) ),
  emuGctLabel_( iConfig.getUntrackedParameter<edm::InputTag>("emuGctInput", edm::InputTag("L1GctEmuDigis") ) ),
  outFilename_( iConfig.getUntrackedParameter<string>("outFile", "gctAnalyzer.txt") ),
  doHW_( iConfig.getUntrackedParameter<bool>("doHardware", true) ),
  doEmu_( iConfig.getUntrackedParameter<bool>("doEmulated", true) ),
  doRctEM_( iConfig.getUntrackedParameter<bool>("doRctEm", true) ),
  doEM_( iConfig.getUntrackedParameter<bool>("doEm", true) ),
  doRegions_( iConfig.getUntrackedParameter<bool>("doRegions", false) ),
  doJets_( iConfig.getUntrackedParameter<bool>("doJets", false) ),
  doInternEM_( iConfig.getUntrackedParameter<bool>("doInternEm", true) ),
  doFibres_( iConfig.getUntrackedParameter<bool>("doFibres", false) ),
  doEnergySums_( iConfig.getUntrackedParameter<bool>("doEnergySums", false) ),
  emMinRank_( iConfig.getUntrackedParameter<unsigned>("emMinRank", 1) ),
  jetMinRank_( iConfig.getUntrackedParameter<unsigned>("jetMinRank", 1) )
{
  //now do what ever initialization is needed

  outFile_.open(outFilename_.c_str(), ios::out);

}


DumpGctDigis::~DumpGctDigis()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  outFile_.close();

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DumpGctDigis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  std::stringstream text;

  text << "Run :" << iEvent.id().run() << "  Event :" << iEvent.id().event() << endl;
  
  // EM
  if (doRctEM_ && doHW_) { doRctEM(iEvent, rawLabel_, text); }
  if (doRctEM_ && doEmu_) { doRctEM(iEvent, emuRctLabel_, text); }
  if (doEM_ && doHW_) { doEM(iEvent, rawLabel_, text); }
  if (doEM_ && doEmu_){ doEM(iEvent, emuGctLabel_, text); }

  // Jets
  if (doRegions_ && doHW_) { doRegions(iEvent, rawLabel_, text); }
  if (doRegions_ && doEmu_) { doRegions(iEvent, emuRctLabel_, text); }
  if (doJets_ && doHW_) { doJets(iEvent, rawLabel_, text); }
  if (doJets_ && doEmu_) { doJets(iEvent, emuGctLabel_, text); }

  // Energy Sums
  if (doEnergySums_ && doHW_) { doEnergySums(iEvent, rawLabel_, text); }
  if (doEnergySums_ && doEmu_) { doEnergySums(iEvent, emuGctLabel_, text); }

  // debugging
  if (doInternEM_ && doHW_) { doInternEM(iEvent, rawLabel_, text); }
  if (doFibres_ && doHW_) { doFibres(iEvent, rawLabel_, text); }

  edm::LogInfo("L1TCaloEvents") << text.str();

}

void DumpGctDigis::doEM(const edm::Event& iEvent, const edm::InputTag& label, std::stringstream& text) {

  using namespace edm;

  Handle<L1GctEmCandCollection> isoEm;
  Handle<L1GctEmCandCollection> nonIsoEm;

  L1GctEmCandCollection::const_iterator ie;
  L1GctEmCandCollection::const_iterator ne;
  
  iEvent.getByLabel(label.label(),"isoEm",isoEm);
  iEvent.getByLabel(label.label(),"nonIsoEm",nonIsoEm);

  text << "Iso EM from : " << label.label() << endl;
  for (ie=isoEm->begin(); ie!=isoEm->end(); ie++) {
    if (ie->rank() >= emMinRank_) {
      text << (*ie) 
	   << " ieta(detID)=" << ie->regionId().ieta()
	   << " iphi(detID)=" << ie->regionId().iphi()
	   << endl;
    } 
  }
  text << endl;
  
  text << "Non-iso EM from : " << label.label() << endl;
  for (ne=nonIsoEm->begin(); ne!=nonIsoEm->end(); ne++) {
    if (ne->rank() >= emMinRank_) {
      text << (*ne) 
	   << " ieta(detID)=" << ne->regionId().ieta()
	   << " iphi(detID)=" << ne->regionId().iphi()
	   << endl;
    } 
  }
  text << endl;

}

void DumpGctDigis::doRctEM(const edm::Event& iEvent, const edm::InputTag& label, std::stringstream& text) {

  using namespace edm;

  Handle<L1CaloEmCollection> em;

  L1CaloEmCollection::const_iterator e;
 
  iEvent.getByLabel(label, em);

  text << "RCT EM from : " << label.label() << endl;
  for (e=em->begin(); e!=em->end(); e++) {
    if (e->rank() >= emMinRank_) {
      text << (*e) 
               << " ieta(detID)=" << e->regionId().ieta()
               << " iphi(detID)=" << e->regionId().iphi()
               << endl;
    }
  } 
  text << endl;
  
}


void DumpGctDigis::doRegions(const edm::Event& iEvent, const edm::InputTag& label, std::stringstream& text) {

  using namespace edm;

  Handle<L1CaloRegionCollection> rgns;

  L1CaloRegionCollection::const_iterator r;
  
  iEvent.getByLabel(label, rgns);

  text << "Regions from : " << label.label() << endl;
  for (r=rgns->begin(); r!=rgns->end(); r++) {
    if (r->et() >= jetMinRank_) {
      text << (*r) << endl;
    }
  } 
  text << endl;

}


void DumpGctDigis::doJets(const edm::Event& iEvent, const edm::InputTag& label, std::stringstream& text) {

  using namespace edm;

  Handle<L1GctJetCandCollection> cenJets;
  Handle<L1GctJetCandCollection> forJets;
  Handle<L1GctJetCandCollection> tauJets;
  
  L1GctJetCandCollection::const_iterator cj;
  L1GctJetCandCollection::const_iterator fj;
  L1GctJetCandCollection::const_iterator tj;
  
  const std::string labelStr = label.label();
  
  iEvent.getByLabel(labelStr,"cenJets",cenJets);
  iEvent.getByLabel(labelStr,"forJets",forJets);
  iEvent.getByLabel(labelStr,"tauJets",tauJets);
  
  text << "Central jets from : " << labelStr << endl;
  for (cj=cenJets->begin(); cj!=cenJets->end(); cj++) {
    if (cj->rank() >= jetMinRank_) {
      text << (*cj) << endl;
    }
  } 
  text << endl;
  
  text << "Forward jets from : " << labelStr << endl;
  for (fj=forJets->begin(); fj!=forJets->end(); fj++) {
    if (fj->rank() >= jetMinRank_) {
      text << (*fj) << endl;
    }
  } 
  text << endl;
  
  text << "Tau jets from : " << labelStr << endl;
  for (tj=tauJets->begin(); tj!=tauJets->end(); tj++) {
    if (tj->rank() >= jetMinRank_) {
      text << (*tj) << endl;
    }
  }
}


void DumpGctDigis::doInternEM(const edm::Event& iEvent, const edm::InputTag& label, std::stringstream& text) {

  using namespace edm;

  Handle<L1GctInternEmCandCollection> em;

  L1GctInternEmCandCollection::const_iterator e;
  
  iEvent.getByLabel(label, em);

  text << "Internal EM from : " << label.label() << endl;
  for (e=em->begin(); e!=em->end(); e++) {
    text << (*e) 
             << " ieta(detID)=" << e->regionId().ieta()
             << " iphi(detID)=" << e->regionId().iphi()
             << endl;
  } 
  text << endl;
  
}


void DumpGctDigis::doFibres(const edm::Event& iEvent, const edm::InputTag& label, std::stringstream& text) {

  using namespace edm;

  Handle<L1GctFibreCollection> fibres;

  L1GctFibreCollection::const_iterator f;
  
  iEvent.getByLabel(label, fibres);

  text << "Fibres from : " << label.label() << endl;
  for (f=fibres->begin(); f!=fibres->end(); f++) {
    text << (*f) << endl;
  } 
  text << endl;
  
}

void DumpGctDigis::doEnergySums(const edm::Event& iEvent, const edm::InputTag& label, std::stringstream& text)
{
  using namespace edm;
  
  Handle<L1GctEtTotalCollection> etTotal;
  Handle<L1GctEtHadCollection> etHad;
  Handle<L1GctEtMissCollection> etMiss;
  Handle<L1GctHtMissCollection> htMiss;
  
  iEvent.getByLabel(label, etTotal);
  iEvent.getByLabel(label, etHad);
  iEvent.getByLabel(label, etMiss);
  iEvent.getByLabel(label, htMiss);
  
  text << "Energy sums from: " << label.label() << endl;
  
  L1GctEtTotalCollection::const_iterator et;
  for (et=etTotal->begin(); et!=etTotal->end(); et++){
    text << *(et) << endl;
  }

  L1GctEtHadCollection::const_iterator ht;
  for (ht=etHad->begin(); ht!=etHad->end(); ht++){
    text << *(ht) << endl;
  }

  L1GctEtMissCollection::const_iterator met;
  for (met=etMiss->begin(); met!=etMiss->end(); met++){
    text << *(met) << endl;
  }

  L1GctHtMissCollection::const_iterator mht;
  for (mht=htMiss->begin(); mht!=htMiss->end(); mht++){
    text << *(mht) << endl;
  }

}
