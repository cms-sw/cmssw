
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctTestAnalyzer.h"

using std::cout;
using std::endl;

//
// constructors and destructor
//
L1GctTestAnalyzer::L1GctTestAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


L1GctTestAnalyzer::~L1GctTestAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1GctTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // get some GCT digis

   Handle<L1GctEmCandCollection> isoEm;
   Handle<L1GctEmCandCollection> nonIsoEm;
   Handle<L1GctJetCandCollection> cenJets;
   Handle<L1GctJetCandCollection> forJets;
   Handle<L1GctJetCandCollection> tauJets;

   L1GctEmCandCollection::const_iterator ie;
   L1GctEmCandCollection::const_iterator ne;
   L1GctJetCandCollection::const_iterator cj;
   L1GctJetCandCollection::const_iterator fj;
   L1GctJetCandCollection::const_iterator tj;

   iEvent.getByType(isoEm);
   iEvent.getByType(nonIsoEm);
   iEvent.getByType(cenJets);
   iEvent.getByType(forJets);
   iEvent.getByType(tauJets);

   cout << "Isolated EM objects :" << endl;
   for (ie=isoEm->begin(); ie!=isoEm->end(); ie++) {
     cout << (*ie) << endl;
   } 
   cout << endl;

   cout << "Non-isolated EM objects :" << endl;
   for (ne=nonIsoEm->begin(); ne!=nonIsoEm->end(); ne++) {
     cout << (*ne) << endl;
   } 
   cout << endl;

   cout << "Central jets :" << endl;
   for (cj=cenJets->begin(); cj!=cenJets->end(); cj++) {
     cout << (*cj) << endl;
   } 
   cout << endl;

   cout << "Forward jets : " << endl;
   for (fj=forJets->begin(); fj!=forJets->end(); fj++) {
     cout << (*fj) << endl;
   } 
   cout << endl;

   cout << "Tau jets :" << endl;
   for (tj=tauJets->begin(); tj!=tauJets->end(); tj++) {
     cout << (*tj) << endl;
   } 
   cout << endl;

}

