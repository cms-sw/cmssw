#include "Fireworks/FWInterface/interface/FWFFHelper.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TColor.h"
#include "TStyle.h"
#include "TEnv.h"
#include "TRint.h"
#include "TEveManager.h"
#include "TEveEventManager.h"
#include "TEveTrackPropagator.h"
#include "TGLWidget.h"
#include "TEveBrowser.h"

#include <cassert>
#include <iostream>

class FWFFTRint : public TRint
{
public:
   FWFFTRint(const char *appClassName, Int_t *argc, char **argv, bool rootPrompt)
    : TRint(appClassName, argc, argv, nullptr, 0, !rootPrompt),
      m_rootPrompt(rootPrompt)
      {
         if (rootPrompt)
            return;
            
         SetPrompt("");
         GetInputHandler()->Remove();
      }

   Bool_t HandleTermInput() override
      {
         if (m_rootPrompt)
            return TRint::HandleTermInput();
         return true;
      }
private:
   bool  m_rootPrompt;
};

FWFFHelper::FWFFHelper(const edm::ParameterSet &ps, const edm::ActivityRegistry &)
   : m_Rint(nullptr)
{
   printf ("CMSSW is starting... You should always have a 2 minutes walk every 45 minutes anyways.\n");
   const char* dummyArgvArray[] = {"cmsRun"};
   char**      dummyArgv = const_cast<char**>(dummyArgvArray);
   int         dummyArgc = 1;

   m_Rint = new FWFFTRint("App", &dummyArgc, dummyArgv, ps.getUntrackedParameter<bool>("rootPrompt"));
   assert(TApplication::GetApplications()->GetSize());

   gROOT->SetBatch(kFALSE);
   TApplication::NeedGraphicsLibs();
 
   try {
      TGLWidget* w = TGLWidget::Create(gClient->GetDefaultRoot(), kTRUE, kTRUE, nullptr, 10, 10);
      delete w;
   }
   catch (std::exception& iException) {
      std::cerr <<"Insufficient GL support. " << iException.what() << std::endl;
      throw;
   }
   
   TEveManager::Create(kFALSE, "FI");
}
