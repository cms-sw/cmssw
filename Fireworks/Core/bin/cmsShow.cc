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

/* NOTE: This is a short term work around until FWLite can properly handle the MessageLogger
 */
#include "FWCore/MessageLogger/interface/AbstractMLscribe.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

namespace {
   class SilentMLscribe : public edm::service::AbstractMLscribe {
      
   public:
      SilentMLscribe() {}
            
      // ---------- member functions ---------------------------
      virtual
      void  runCommand(edm::MessageLoggerQ::OpCode  opcode, void * operand);
      
   private:
      SilentMLscribe(const SilentMLscribe&); // stop default
      
      const SilentMLscribe& operator=(const SilentMLscribe&); // stop default
      
      // ---------- member data --------------------------------
      
   };      
   
   void  
   SilentMLscribe::runCommand(edm::MessageLoggerQ::OpCode  opcode, void * operand) {
      //even though we don't print, have to clean up memory
      switch (opcode) {
         case edm::MessageLoggerQ::LOG_A_MESSAGE: {
            edm::ErrorObj *  errorobj_p = static_cast<edm::ErrorObj *>(operand);
            delete errorobj_p;
            break;
         }
         case edm::MessageLoggerQ::JOBREPORT:
         case edm::MessageLoggerQ::JOBMODE:
         case edm::MessageLoggerQ::GROUP_STATS:
            std::string* string_p = static_cast<std::string*> (operand);
            delete string_p;
            break;
         default:
            break;
      }
   }   
}

void run_app(TApplication &app, int argc, char **argv)
{
   //Remove when FWLite handles the MessageLogger
   SilentMLscribe scribe;
   edm::MessageLoggerQ::setMLscribe_ptr(&scribe);
   edm::MessageDrop::instance()->messageLoggerScribeIsRunning = edm::MLSCRIBE_RUNNING_INDICATOR;
   //---------------------
   AutoLibraryLoader::enable();
   std::auto_ptr<CmsShowMain> pMain( new CmsShowMain(argc,argv) );
   app.Run();
   pMain.reset();

   TEveManager::Terminate();
   app.Terminate();

   //Remove when FWLite handled the MessageLogger
   edm::MessageLoggerQ::MLqEND();
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
