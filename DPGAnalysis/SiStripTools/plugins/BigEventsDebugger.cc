// -*- C++ -*-
//
// Package:    BigEventsDebugger
// Class:      BigEventsDebugger
// 
/**\class BigEventsDebugger BigEventsDebugger.cc myTKAnalyses/BigEventsDebugger/src/BigEventsDebugger.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Sun Nov 16 16:04:44 CET 2008
// $Id: BigEventsDebugger.cc,v 1.1 2011/10/02 16:58:33 venturia Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include "TH2F.h"
#include "TProfile.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DPGAnalysis/SiStripTools/interface/DigiCollectionProfiler.h"

//
// class decleration
//

class BigEventsDebugger : public edm::EDAnalyzer {
   public:
      explicit BigEventsDebugger(const edm::ParameterSet&);
      ~BigEventsDebugger();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  bool m_singleevents;

  TProfile* m_tibprof;
  TProfile* m_tobprof;
  TProfile* m_tecpprof;
  TProfile* m_tecmprof;

  TH2F* m_tib2d;
  TH2F* m_tob2d;
  TH2F* m_tecp2d;
  TH2F* m_tecm2d;

  std::vector<unsigned int> m_maskedmod;
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
BigEventsDebugger::BigEventsDebugger(const edm::ParameterSet& iConfig):
  m_singleevents(iConfig.getParameter<bool>("singleEvents")),
  m_maskedmod(iConfig.getUntrackedParameter<std::vector<unsigned int> >("maskedModules"))

{
   //now do what ever initialization is needed

  sort(m_maskedmod.begin(),m_maskedmod.end());

  edm::Service<TFileService> tfserv;

  if(!m_singleevents) {
   char dirname[500];
   sprintf(dirname,"Summary");
   TFileDirectory subd = tfserv->mkdir(dirname);
   
   //book histos

   m_tibprof = subd.make<TProfile>("tibprof","TIB Digi charge profile",256,-0.5,255.5);
   m_tobprof = subd.make<TProfile>("tobprof","TOB Digi charge profile",256,-0.5,255.5);
   m_tecpprof = subd.make<TProfile>("tecpprof","TECp Digi charge profile",256,-0.5,255.5);
   m_tecmprof = subd.make<TProfile>("tecmprof","TECm Digi charge profile",256,-0.5,255.5);

   m_tib2d = subd.make<TH2F>("tib2d","TIB Digi charge distribution",256,-0.5,255.5,257,-0.5,256.5);
   m_tob2d = subd.make<TH2F>("tob2d","TOB Digi charge distribution",256,-0.5,255.5,257,-0.5,256.5);
   m_tecp2d = subd.make<TH2F>("tecp2d","TECp Digi charge distribution",256,-0.5,255.5,257,-0.5,256.5);
   m_tecm2d = subd.make<TH2F>("tecm2d","TECm Digi charge distribution",256,-0.5,255.5,257,-0.5,256.5);

  }

}


BigEventsDebugger::~BigEventsDebugger()
{

 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
BigEventsDebugger::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::Service<TFileService> tfserv;

   // create a folder for each event

   if(m_singleevents) {

     char dirname[500];
     sprintf(dirname,"event_%u_%u",iEvent.run(),iEvent.id().event());
     TFileDirectory subd = tfserv->mkdir(dirname);
     
     //book histos
     
     m_tibprof = subd.make<TProfile>("tibprof","TIB Digi charge profile",256,-0.5,255.5);
     m_tobprof = subd.make<TProfile>("tobprof","TOB Digi charge profile",256,-0.5,255.5);
     m_tecpprof = subd.make<TProfile>("tecpprof","TECp Digi charge profile",256,-0.5,255.5);
     m_tecmprof = subd.make<TProfile>("tecmprof","TECm Digi charge profile",256,-0.5,255.5);
     
     m_tib2d = subd.make<TH2F>("tib2d","TIB Digi charge distribution",256,-0.5,255.5,257,-0.5,256.5);
     m_tob2d = subd.make<TH2F>("tob2d","TOB Digi charge distribution",256,-0.5,255.5,257,-0.5,256.5);
     m_tecp2d = subd.make<TH2F>("tecp2d","TECp Digi charge distribution",256,-0.5,255.5,257,-0.5,256.5);
     m_tecm2d = subd.make<TH2F>("tecm2d","TECm Digi charge distribution",256,-0.5,255.5,257,-0.5,256.5);

   }

   //analyze event

   DigiCollectionProfiler profiler(m_tibprof,
				   m_tobprof,
				   m_tecpprof,
				   m_tecmprof,
				   m_tib2d,
				   m_tob2d,
				   m_tecp2d,
				   m_tecm2d);

   Handle<edm::DetSetVector<SiStripDigi> > digis;
   iEvent.getByLabel("siStripDigis","ZeroSuppressed",digis);
   profiler.setMaskedModules(m_maskedmod);
   profiler.analyze(digis);

}


// ------------ method called once each job just before starting event loop  ------------
void 
BigEventsDebugger::beginJob(const edm::EventSetup&)
{
  edm::LogInfo("MaskedModules") << m_maskedmod.size() << " masked modules ";
  for(std::vector<unsigned int>::const_iterator it=m_maskedmod.begin();it!=m_maskedmod.end();it++) {
    edm::LogVerbatim("MaskedModules") << (*it);
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BigEventsDebugger::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(BigEventsDebugger);
