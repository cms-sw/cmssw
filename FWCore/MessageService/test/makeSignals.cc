#include "FWCore/MessageService/test/makeSignals.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>
#include <csignal>

#define RAISE_SEGV
//#define RAISE_USR2

namespace edmtest
{

void
  makeSignals::analyze( edm::Event      const & e
                       ,edm::EventSetup const & /*unused*/
                      )
{
#ifdef RAISE_SEGV
  int signum = 11;
  std::string SigName("SIGSEGV");
#endif

#ifdef RAISE_USR2
  int signum = 12;
  std::string SigName("SIGUSR2");
#endif
  edm::MessageDrop::instance()->debugEnabled  = true;

       LogTrace    ("cat_A") << "LogTrace was used to send this mess" << "age";
       LogDebug    ("cat_B") << "LogDebug was used to send this other message";
  edm::LogVerbatim ("cat_A") << "LogVerbatim was us" << "ed to send this message";
  if( edm::isInfoEnabled() ) 
     edm::LogInfo  ("cat_B") << "LogInfo was used to send this other message\n" ;

  if( e.id().event() == 5 )
   {
    std::cerr << "Raising Signal " << SigName << " = " << signum << std::endl;
    edm::LogInfo("Signals") << "Raising Signal " << SigName << " = " << signum ;
#ifdef RAISE_SEGV
    raise(SIGSEGV);
#endif

#ifdef RAISE_USR2
    raise(SIGUSR2);
#endif

//  Force a Seg Fault
//  int * pint = 0;
//  int rint = *pint;
   }
}  // makeSignals::analyze()
}  // namespace edmtest


using edmtest::makeSignals;
DEFINE_FWK_MODULE(makeSignals);
