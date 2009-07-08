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
#include <string.h>
#include <memory>

void run_app(TApplication &app, int argc, char **argv)
{
   AutoLibraryLoader::enable();
   std::auto_ptr<CmsShowMain> pMain( new CmsShowMain(argc,argv) );
   app.Run();
   pMain.reset();

   TEveManager::Terminate();
   app.Terminate();
}

int main (int argc, char **argv)
{
   char* dummyArgv[] = {"cmsShow"};
   int dummyArgc = 1;
   gEnv->SetValue("Gui.BackgroundColor", "#9f9f9f");

   // check root interactive promp
   bool isri = false;
   for (Int_t i =0; i<argc; i++)
   {
      if (strncmp(argv[i], "-r", 2) == 0||
	  strncmp(argv[i], "--root", 6) == 0)
      {
         isri=true;
         break;
      }
   }

   if (isri) {
      std::cout<<""<<std::endl;
      std::cout<<"WARNING:You are running cmsShow with ROOT prompt enabled."<<std::endl;
      std::cout<<"If you encounter an issue you suspect to be a bug in     "<<std::endl;
      std::cout<<"cmsShow, please re-run without this option and try to    "<<std::endl;
      std::cout<<"reproduce it before submitting a bug-report.             "<<std::endl;
      std::cout<<""<<std::endl;

      TRint app("cmsShow", &dummyArgc, dummyArgv);
      run_app(app,argc, argv);
   } else {
      std::cout <<"starting"<<std::endl;
      TApplication app("cmsShow", &dummyArgc, dummyArgv); 
      run_app(app, argc, argv);
   }

   return 0;
}
