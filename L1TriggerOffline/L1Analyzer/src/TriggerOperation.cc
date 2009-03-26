// -*- C++ -*-
//
// Package:    TriggerOperation
// Class:      TriggerOperation
// 
/**\class TriggerOperation TriggerOperation.cc Demo/TriggerOperation/src/TriggerOperation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Georgia KARAPOSTOLI
//         Created:  Wed Feb 11 18:34:11 CET 2009
// $Id$
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1TriggerOffline/L1Analyzer/interface/SimpleHBits.h"
//
// class decleration
//

using namespace std;

class TriggerOperation : public edm::EDAnalyzer {
   public:
      explicit TriggerOperation(const edm::ParameterSet&);
      ~TriggerOperation();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  edm::InputTag m_l1GtReadoutRecord;

//   /// trigger masks
//   const L1GtTriggerMask* m_l1GtTmAlgo;
//   unsigned long long m_l1GtTmAlgoCacheID;
  
//   const L1GtTriggerMask* m_l1GtTmTech;
//   unsigned long long m_l1GtTmTechCacheID;
  
//   std::vector<unsigned int> m_triggerMaskAlgoTrig;
//   std::vector<unsigned int> m_triggerMaskTechTrig;
  
  SimpleHBits *m_TriggerBits;

  // vector<int> nBits;

      // ----------member data ---------------------------
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
TriggerOperation::TriggerOperation(const edm::ParameterSet& iConfig) :
  m_l1GtReadoutRecord(iConfig.getUntrackedParameter<edm::InputTag>("L1GtReadoutRecordTag"))
{
   //now do what ever initialization is needed
  m_TriggerBits = new SimpleHBits("TriggerBits",iConfig);

   // initialize cached IDs
    
  //   m_l1GtTmAlgoCacheID = 0;
//     m_l1GtTmTechCacheID = 0;

  // m_l1GtReadoutRecord(iConfig.getUntrackedParameter<edm::InputTag>("L1GtReadoutRecordTag"));
}


TriggerOperation::~TriggerOperation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete m_TriggerBits;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TriggerOperation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  // get L1GlobalTriggerReadoutRecord
   edm::Handle<L1GlobalTriggerReadoutRecord> gtRecord;
   iEvent.getByLabel(m_l1GtReadoutRecord, gtRecord);
   
   if (!gtRecord.isValid()) {
     
     LogDebug("L1GlobalTriggerRecordProducer")
       << "\n\n Error: no L1GlobalTriggerReadoutRecord found with input tag "
       << m_l1GtReadoutRecord
       << "\n Returning empty L1GlobalTriggerRecord.\n\n"
       << std::endl;
     
     return;
   }
   
   DecisionWord algoDecisionWord = gtRecord->decisionWord();    
   TechnicalTriggerWord techDecisionWord = gtRecord->technicalTriggerWord();   
   
   int tBit=0;
   
   for (std::vector<bool>::iterator 
	  itBit = algoDecisionWord.begin(); itBit != algoDecisionWord.end(); ++itBit) {
     bool algoTrigger = algoDecisionWord.at(tBit);
     if (algoTrigger) {m_TriggerBits->FillTB(static_cast<float>(tBit));}
     tBit++;
   }
   
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
TriggerOperation::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TriggerOperation::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TriggerOperation);
