#include "Fireworks/FWInterface/interface/FWFFHelper.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#define private public
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
    : TRint(appClassName, argc, argv, 0, 0, !rootPrompt),
      m_rootPrompt(rootPrompt)
      {
         if (rootPrompt)
            return;
            
         SetPrompt("");
         fInputHandler->Remove();
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
   : m_Rint(0)
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
      TGLWidget* w = TGLWidget::Create(gClient->GetDefaultRoot(), kTRUE, kTRUE, 0, 10, 10);
      delete w;
   }
   catch (std::exception& iException) {
      std::cerr <<"Insufficient GL support. " << iException.what() << std::endl;
      throw;
   }

// AMT workaround for an agressive clenup in 5.43.18
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,34,18)
   if (!gStyle) {
      TColor::fgInitDone=false;
      TColor::InitializeColors();
      TStyle::BuildStyles();
      gROOT->SetStyle(gEnv->GetValue("Canvas.Style", "Modern"));
      gStyle = gROOT->GetStyle("Classic");
   }
#endif

    TEveManager::Create(kFALSE, "FI");
}
