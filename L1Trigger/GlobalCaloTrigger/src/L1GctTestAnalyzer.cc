
// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctTestAnalyzer.h"

using std::string;
using std::ios;
using std::endl;

//
// constructors and destructor
//
L1GctTestAnalyzer::L1GctTestAnalyzer(const edm::ParameterSet& iConfig) :
  rawLabel_( iConfig.getUntrackedParameter<string>("rawLabel", "L1GctRawDigis") ),
  emuLabel_( iConfig.getUntrackedParameter<string>("emuLabel", "L1GctEmuDigis") ),
  outFilename_( iConfig.getUntrackedParameter<string>("outFile", "gctAnalyzer.txt") ),
  doRctEM_( iConfig.getUntrackedParameter<unsigned>("doRctEm", 1) ),
  doInternEM_( iConfig.getUntrackedParameter<unsigned>("doInternEm", 1) ),
  doEM_( iConfig.getUntrackedParameter<unsigned>("doEm", 1) ),
  doJets_( iConfig.getUntrackedParameter<unsigned>("doJets", 0) )
{
  //now do what ever initialization is needed

  outFile_.open(outFilename_.c_str(), ios::out);

}


L1GctTestAnalyzer::~L1GctTestAnalyzer()
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
L1GctTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   if (doRctEM_!=0) {
     doRctEM(iEvent, rawLabel_);
     //doRctEM(iEvent, emuLabel_);
   }
   if (doInternEM_!=0) {
     doInternEM(iEvent, rawLabel_);
     //doInternEM(iEvent, emuLabel_);
   }
   if (doEM_!=0) {
     doEM(iEvent, rawLabel_);
     doEM(iEvent, emuLabel_);
   }

   if (doJets_!=0) {
     doJets(iEvent, rawLabel_);
     doJets(iEvent, emuLabel_);
   }
}

void L1GctTestAnalyzer::doEM(const edm::Event& iEvent, string label) {

  using namespace edm;

  Handle<L1GctEmCandCollection> isoEm;
  Handle<L1GctEmCandCollection> nonIsoEm;

  L1GctEmCandCollection::const_iterator ie;
  L1GctEmCandCollection::const_iterator ne;
  
  iEvent.getByLabel(label,"isoEm",isoEm);
  iEvent.getByLabel(label,"nonIsoEm",nonIsoEm);

  outFile_ << "From : " << label << endl;

  outFile_ << "Iso EM :" << endl;
  for (ie=isoEm->begin(); ie!=isoEm->end(); ie++) {
    outFile_ << (*ie) << endl;
  } 
  outFile_ << endl;
  
  outFile_ << "Non-iso EM :" << endl;
  for (ne=nonIsoEm->begin(); ne!=nonIsoEm->end(); ne++) {
    outFile_ << (*ne) << endl;
  } 
  outFile_ << endl;

}

void L1GctTestAnalyzer::doRctEM(const edm::Event& iEvent, string label) {

  using namespace edm;

  Handle<L1CaloEmCollection> em;

  L1CaloEmCollection::const_iterator e;
 
  iEvent.getByLabel(label, "", em);

  outFile_ << "From : " << label << endl;

  outFile_ << "RCT EM :" << endl;
  for (e=em->begin(); e!=em->end(); e++) {
    outFile_ << (*e) << endl;
  } 
  outFile_ << endl;
  
}


void L1GctTestAnalyzer::doInternEM(const edm::Event& iEvent, string label) {

  using namespace edm;

  Handle<L1GctInternEmCandCollection> em;

  L1GctInternEmCandCollection::const_iterator e;
  
  iEvent.getByLabel(label, "", em);

  outFile_ << "From : " << label << endl;

  outFile_ << "Internal EM :" << endl;
  for (e=em->begin(); e!=em->end(); e++) {
    outFile_ << (*e) << endl;
  } 
  outFile_ << endl;
  
}



void L1GctTestAnalyzer::doJets(const edm::Event& iEvent, string label) {

  using namespace edm;

  Handle<L1GctJetCandCollection> cenJets;
  Handle<L1GctJetCandCollection> forJets;
  Handle<L1GctJetCandCollection> tauJets;
  
  L1GctJetCandCollection::const_iterator cj;
  L1GctJetCandCollection::const_iterator fj;
  L1GctJetCandCollection::const_iterator tj;
  
  
  iEvent.getByLabel(label,cenJets);
  iEvent.getByLabel(label,"forJets",forJets);
  iEvent.getByLabel(label,"tauJets",tauJets);
  
  outFile_ << "Central jets :" << endl;
  for (cj=cenJets->begin(); cj!=cenJets->end(); cj++) {
    outFile_ << (*cj) << endl;
  } 
  outFile_ << endl;
  
  outFile_ << "Forward jets : " << endl;
  for (fj=forJets->begin(); fj!=forJets->end(); fj++) {
    outFile_ << (*fj) << endl;
  } 
  outFile_ << endl;
  
  outFile_ << "Tau jets :" << endl;
  for (tj=tauJets->begin(); tj!=tauJets->end(); tj++) {
    outFile_ << (*tj) << endl;
  } 
  outFile_ << endl;
  

}
