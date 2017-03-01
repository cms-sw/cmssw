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
#include "TH2F.h"
#include "TProfile.h"
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

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
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endJob() override ;

      // ----------member data ---------------------------

  edm::EDGetTokenT<L1AcceptBunchCrossingCollection> m_l1abccollectionToken;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;

  RunHistogramManager m_rhm;

  TH2F** m_hoffsets;
  TProfile** m_horboffvsorb;
  TProfile** m_hbxoffvsorb;


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
  m_l1abccollectionToken(consumes<L1AcceptBunchCrossingCollection>(iConfig.getParameter<edm::InputTag>("l1ABCCollection"))),
  m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin",250)),
  m_LSfrac(iConfig.getUntrackedParameter<unsigned int>("startingLSFraction",16)),
  m_rhm(consumesCollector())
{
   //now do what ever initialization is needed

  m_hoffsets = m_rhm.makeTH2F("offsets","Orbit vs BX offsets between SCAL and Event",2*3564+1,-3564.5,3564.5,201,-100.5,100.5);
  m_horboffvsorb = m_rhm.makeTProfile("orboffvsorb","SCAL Orbit offset vs orbit number",m_LSfrac*m_maxLS,0,m_maxLS*262144);
  m_hbxoffvsorb = m_rhm.makeTProfile("bxoffvsorb","SCAL BX offset vs orbit number",m_LSfrac*m_maxLS,0,m_maxLS*262144);

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
   iEvent.getByToken(m_l1abccollectionToken,pIn);

   // offset computation
       for(L1AcceptBunchCrossingCollection::const_iterator l1abc=pIn->begin();l1abc!=pIn->end();++l1abc) {
	 if(l1abc->l1AcceptOffset()==0) {
	   if(m_hoffsets && *m_hoffsets) 
	     (*m_hoffsets)->Fill((int)l1abc->bunchCrossing()-(int)iEvent.bunchCrossing(),
				 (long long)l1abc->orbitNumber()-(long long)iEvent.orbitNumber());
	   if(m_horboffvsorb && *m_horboffvsorb) 
	     (*m_horboffvsorb)->Fill(iEvent.orbitNumber(),(long long)l1abc->orbitNumber()-(long long)iEvent.orbitNumber());
	   if(m_hbxoffvsorb && *m_hbxoffvsorb) 
	     (*m_hbxoffvsorb)->Fill(iEvent.orbitNumber(),(int)l1abc->bunchCrossing()-(int)iEvent.bunchCrossing());
	 }
       }


   // dump of L1ABC collection

   edm::LogInfo("L1ABCDebug") << "Dump of L1AcceptBunchCrossing Collection for event in orbit " 
			      << iEvent.orbitNumber() << " and BX " << iEvent.bunchCrossing();

   for(L1AcceptBunchCrossingCollection::const_iterator l1abc=pIn->begin();l1abc!=pIn->end();++l1abc) {
     edm::LogVerbatim("L1ABCDebug") << *l1abc;
   }

}

void
L1ABCDebugger::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {

  m_rhm.beginRun(iRun);

  if(m_hoffsets && *m_hoffsets) {
    (*m_hoffsets)->GetXaxis()->SetTitle("#Delta BX (SCAL-Event)");    (*m_hoffsets)->GetYaxis()->SetTitle("#Delta orbit (SCAL-Event)");
  }
  if(m_horboffvsorb && *m_horboffvsorb) {
    (*m_horboffvsorb)->GetXaxis()->SetTitle("Orbit");    (*m_horboffvsorb)->GetYaxis()->SetTitle("#Delta orbit (SCAL-Event)");
    (*m_horboffvsorb)->SetCanExtend(TH1::kXaxis);
  }
  if(m_hbxoffvsorb && *m_hbxoffvsorb) {
    (*m_hbxoffvsorb)->GetXaxis()->SetTitle("Orbit");    (*m_hbxoffvsorb)->GetYaxis()->SetTitle("#Delta BX (SCAL-Event)");
    (*m_hbxoffvsorb)->SetCanExtend(TH1::kXaxis);
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
