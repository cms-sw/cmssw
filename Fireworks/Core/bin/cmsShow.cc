//#define  FW_ROOT_INTERACTIVE

#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "Rtypes.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TSystem.h" 
#include "TGLSAViewer.h"
#include "TEveManager.h"
#include "TRint.h"
#include "TApplication.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include <iostream>
#include <memory>

int main (int argc, char **argv)
{
   std::cout <<" starting"<<std::endl;
   char* dummyArgv[] = {"cmsShow"};
   int dummyArgc = 1;
   gEnv->SetValue("Gui.BackgroundColor", "#9f9f9f");
#ifdef FW_ROOT_INTERACTIVE
    TRint app("cmsShow", &dummyArgc, dummyArgv);
#else
    TApplication app("cmsShow", &dummyArgc, dummyArgv);
#endif
   AutoLibraryLoader::enable();
   std::auto_ptr<CmsShowMain> pMain( new CmsShowMain(argc,argv) );
   app.Run();
   pMain.reset();
   //dynamic_cast<TGLSAViewer*>(gEve->GetGLViewer())->DeleteMenuBar(); //CDJ: Off for now
   TEveManager::Terminate();

   //the handler has a pointer back to TApplication so must be removed
   TFileHandler* handler = gSystem->RemoveFileHandler(gXDisplay);
   if(0!=handler) {gXDisplay=0;}
   delete handler;
   return 0;
}
