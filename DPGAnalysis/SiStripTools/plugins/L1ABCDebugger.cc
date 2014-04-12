// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      L1ABCDebugger
//
/**\class L1ABCDebugger L1ABCDebugger.cc DPGAnalysis/SiStripTools/plugins/L1ABCDebugger.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Jul 19 11:56:00 CEST 2009
//
//


// system include files
#include <memory>

// user include files
#include "TH1F.h"
#include "TProfile.h"
#include <vector>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
//
// class decleration
//

class L1ABCDebugger : public edm::EDAnalyzer {
 public:
    explicit L1ABCDebugger(const edm::ParameterSet&);
    ~L1ABCDebugger();


   private:
      virtual void beginJob() override ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      // ----------member data ---------------------------

  edm::EDGetTokenT<L1AcceptBunchCrossingCollection> _l1abccollectionToken;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1ABCDebugger::L1ABCDebugger(const edm::ParameterSet& iConfig):
  _l1abccollectionToken(consumes<L1AcceptBunchCrossingCollection>(iConfig.getParameter<edm::InputTag>("l1ABCCollection")))
{
   //now do what ever initialization is needed

}


L1ABCDebugger::~L1ABCDebugger()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1ABCDebugger::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<L1AcceptBunchCrossingCollection > pIn;
   iEvent.getByToken(_l1abccollectionToken,pIn);

   // offset computation

   edm::LogInfo("L1ABCDebug") << "Dump of L1AcceptBunchCrossing Collection";

   for(L1AcceptBunchCrossingCollection::const_iterator l1abc=pIn->begin();l1abc!=pIn->end();++l1abc) {
     edm::LogVerbatim("L1ABCDebug") << *l1abc;
   }

}

// ------------ method called once each job just before starting event loop  ------------
void
L1ABCDebugger::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void
L1ABCDebugger::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(L1ABCDebugger);
