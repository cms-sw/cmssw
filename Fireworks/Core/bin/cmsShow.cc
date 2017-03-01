#include "TEnv.h"
#include "TSystem.h" 
#include "TEveManager.h"
#include "TRint.h"
#include "TApplication.h"
#include "TSysEvtHandler.h"
#include "Getline.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <memory>
#include <signal.h>

#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/fwPaths.h"
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
      void  runCommand(edm::MessageLoggerQ::OpCode  opcode, void * operand) override;
      
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
            //std::cerr<<errorobj_p->xid().severity.getInputStr()<<" "<<errorobj_p->xid().id<<" -----------------------"<<std::endl;
            //std::cerr <<errorobj_p->fullText()<<std::endl;
            delete errorobj_p;
            break;
         }
         case edm::MessageLoggerQ::JOBMODE:
         case edm::MessageLoggerQ::GROUP_STATS:
         {
            std::string* string_p = static_cast<std::string*> (operand);
            delete string_p;
            break;
         }
         default:
            break;
      }
   }   
   
}

namespace
{
void signal_handler_wrapper(int sid, siginfo_t* sinfo, void* sctx)
{
#if defined(R__LINUX)
   std::cerr << "Program received signal ID = " <<  sid << std::endl;
   std::cerr << "Printing stack trace ... " << std::endl;

   TString gdbCommand("scripts/gdb-backtrace.sh");
   fireworks::setPath(gdbCommand); 
   gdbCommand += " ";

   gdbCommand += gSystem->GetPid();

   gSystem->Exec(gdbCommand.Data());
   gSystem->Exit(sid);   
   Getlinem(kCleanUp, 0);
#endif
}
}

void run_app(TApplication &app, int argc, char **argv)
{
   //Remove when FWLite handles the MessageLogger
   edm::MessageLoggerQ::setMLscribe_ptr(std::shared_ptr<edm::service::AbstractMLscribe>(std::make_shared<SilentMLscribe>()));
   edm::MessageDrop::instance()->messageLoggerScribeIsRunning = edm::MLSCRIBE_RUNNING_INDICATOR;
   //---------------------
   std::auto_ptr<CmsShowMain> pMain( new CmsShowMain(argc,argv) );

   // Avoid haing root handling various associated to an error and install 
   // back the default ones.
   gSystem->ResetSignal(kSigBus);
   gSystem->ResetSignal(kSigSegmentationViolation);
   gSystem->ResetSignal(kSigIllegalInstruction);
   gSystem->ResetSignal(kSigSystem);
   gSystem->ResetSignal(kSigPipe);
   gSystem->ResetSignal(kSigFloatingException);
   
   struct sigaction sac;
   sac.sa_sigaction = signal_handler_wrapper;
   sigemptyset(&sac.sa_mask);
   sac.sa_flags = SA_SIGINFO;
   sigaction(SIGILL,  &sac, 0);
   sigaction(SIGSEGV, &sac, 0);
   sigaction(SIGBUS,  &sac, 0);
   sigaction(SIGFPE,  &sac, 0);

   app.Run();
   pMain.reset();

   TEveManager::Terminate();
   app.Terminate();

   //Remove when FWLite handled the MessageLogger
   edm::MessageLoggerQ::MLqEND();
}

int main (int argc, char **argv)
{
   const char* dummyArgvArray[] = {"cmsShow"};
   char** dummyArgv = const_cast<char**>(dummyArgvArray);
   int dummyArgc = 1;
   gEnv->SetValue("Gui.BackgroundColor", "#9f9f9f");

   // print version
   TString infoText;
   if (gSystem->Getenv("CMSSW_VERSION"))
   {
      infoText = gSystem->Getenv("CMSSW_VERSION");
   }
   else
   {
      TString infoFileName("data/version.txt");
      fireworks::setPath(infoFileName);
      std::ifstream infoFile(infoFileName);
      infoText.ReadLine(infoFile);
      infoFile.close();
   }
   printf("Starting cmsShow, version %s.\n", infoText.Data());
   fflush(stdout);

   // check root interactive promp
   bool isri = false;
   for (Int_t i =0; i<argc; i++)
   {
      if (strncmp(argv[i], "-r", 2) == 0 ||
	  strncmp(argv[i], "--root", 6) == 0)
      {
         isri=true;
      }
   }

   try {
      if (isri) {
         std::cerr<<""<<std::endl;
         std::cerr<<"WARNING:You are running cmsShow with ROOT prompt enabled."<<std::endl;
         std::cerr<<"If you encounter an issue you suspect to be a bug in     "<<std::endl;
         std::cerr<<"cmsShow, please re-run without this option and try to    "<<std::endl;
         std::cerr<<"reproduce it before submitting a bug-report.             "<<std::endl;
         std::cerr<<""<<std::endl;

         TRint app("cmsShow", &dummyArgc, dummyArgv);
         run_app(app,argc, argv);
      } else {
         TApplication app("cmsShow", &dummyArgc, dummyArgv);
         run_app(app,argc, argv);
      }
   }
   catch(std::exception& iException)
   {
      std::cerr <<"CmsShow unhandled exception "<<iException.what()<<std::endl;
      return 1;      
   }

   return 0;
}
