#include "Fireworks/FWInterface/interface/FWFFHelper.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TRint.h"
#include "TEveManager.h"
#include "TEveEventManager.h"
#include "TEveTrackPropagator.h"
#include "TGLWidget.h"
#include "TEveBrowser.h"

#include <cassert>
#include <iostream>

FWFFHelper::FWFFHelper(const edm::ParameterSet &ps, const edm::ActivityRegistry &)
   : m_Rint(0)
{
   printf("FWFFHelper::FWFFHelper CTOR\n");

   std::cout <<" gApplication "<< gApplication <<std::endl;
   std::cout <<" is batch "    << gROOT->IsBatch() <<std::endl;
   std::cout <<" display "     << gSystem->Getenv("DISPLAY") <<std::endl;

   const char* dummyArgvArray[] = {"cmsRun"};
   char**      dummyArgv = const_cast<char**>(dummyArgvArray);
   int         dummyArgc = 1;

   m_Rint = new TRint("App", &dummyArgc, dummyArgv);
   assert(TApplication::GetApplications()->GetSize());

   gROOT->SetBatch(kFALSE);
   std::cout<<"calling NeedGraphicsLibs()"<<std::endl;
   TApplication::NeedGraphicsLibs();
 
   try {
      TGLWidget* w = TGLWidget::Create(gClient->GetDefaultRoot(), kTRUE, kTRUE, 0, 10, 10);
      delete w;
   }
   catch (std::exception& iException) {
      std::cerr <<"Insufficient GL support. " << iException.what() << std::endl;
      throw;
   }
   

   TEveManager::Create(kFALSE, "FIV");
}
