// -*- C++ -*-
//
// Package:     Fireworks/Eve
// Class  :     EveService
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Matevz Tadel
//         Created:  Fri Jun 25 18:57:39 CEST 2010
// $Id$
//

// system include files

// user include files
#include "Fireworks/Eve/interface/EveService.h"

// #include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TRint.h"
#include "TEveManager.h"
#include "TEveEventManager.h"

DEFINE_FWK_SERVICE(EveService);

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//==============================================================================
// constructors and destructor
//==============================================================================

EveService::EveService(const edm::ParameterSet&, edm::ActivityRegistry& ar) :
   m_EveManager(0), m_Rint(0)
{
   printf("EveService::EveService CTOR\n");

   std::cout <<" is batch "<<gROOT->IsBatch()<<std::endl;
   std::cout <<" display "<<gSystem->Getenv("DISPLAY")<<std::endl;

   const char* dummyArgvArray[] = {"cmsRun"};
   char**      dummyArgv = const_cast<char**>(dummyArgvArray);
   int         dummyArgc = 1;

   m_Rint = new TRint("App", &dummyArgc, dummyArgv);
   assert(TApplication::GetApplications()->GetSize());
  
   gROOT->SetBatch(kFALSE);
   std::cout<<"calling NeedGraphicsLibs()"<<std::endl;
   TApplication::NeedGraphicsLibs();

   m_EveManager = TEveManager::Create();

   ar.watchPostBeginJob(this, &EveService::postBeginJob);
   ar.watchPostEndJob  (this, &EveService::postEndJob);

   ar.watchPostProcessEvent(this, &EveService::postProcessEvent);

}

EveService::~EveService()
{
   printf("EveService::~EveService DTOR\n");
}


//==============================================================================
// Service watchers
//==============================================================================

void EveService::postBeginJob()
{
   printf("EveService::postBeginJob\n");

   gSystem->ProcessEvents();
}

void EveService::postEndJob()
{
   printf("EveService::postEndJob\n");

   TEveManager::Terminate();
}

void EveService::postProcessEvent(const edm::Event&, const edm::EventSetup&)
{
   printf("EveService::postProcessEvent: Starting GUI loop.\n");
   printf("Type .q to go to next event, .qqqqq to quit.\n");

   gEve->Redraw3D();

   m_Rint->Run(kTRUE);

   gEve->GetCurrentEvent()->DestroyElements();
}

//==============================================================================
// Data registrators
//==============================================================================

TEveManager* EveService::getManager()
{
   gEve = m_EveManager;
   return m_EveManager;
}

//
// const member functions
//

//
// static member functions
//
