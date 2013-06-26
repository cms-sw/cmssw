#include "DQMServices/Core/interface/Standalone.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "classlib/utils/DebugAids.h"
#include "classlib/utils/Signal.h"
#include "TROOT.h"
#include <iostream>
#include <cstdlib>
#include <errno.h>

#ifndef _NSIG
#define _NSIG NSIG
#endif

static const int FATAL_OPTS = (lat::Signal::FATAL_DEFAULT
			       & ~(lat::Signal::FATAL_ON_INT
				   | lat::Signal::FATAL_ON_QUIT
				   | lat::Signal::FATAL_DUMP_CORE));

// -------------------------------------------------------------------
// Always abort on assertion failure.
static char
onAssertFail(const char *message)
{
  std::cout.flush(); fflush(stdout);
  std::cerr.flush(); fflush(stderr);
  std::cerr << message << "ABORTING\n";
  return 'a';
}

// -------------------------------------------------------------------
// Main program.
int main(int argc, char **argv)
{
  // Install base debugging support.
  lat::DebugAids::failHook(&onAssertFail);
  lat::Signal::handleFatal(argv[0], IOFD_INVALID, 0, 0, FATAL_OPTS);

  // Re-capture signals from ROOT after ROOT has initialised.
  ROOT::GetROOT();
  for (int sig = 1; sig < _NSIG; ++sig) lat::Signal::revert(sig);
  lat::Signal::handleFatal(argv[0], IOFD_INVALID, 0, 0, FATAL_OPTS);

  // Check command line arguments.
  char *output = (argc > 1 ? argv[1] : 0);
  if (! output)
  {
    std::cerr << "Usage: " << argv[0]
	      << " OUTPUT-FILE FILE...\n";
    return 1;
  }

  // Process each file given as argument.
  edm::ParameterSet emptyps;
  std::vector<edm::ParameterSet> emptyset;
  edm::ServiceToken services(edm::ServiceRegistry::createSet(emptyset));	 
  edm::ServiceRegistry::Operate operate(services);	 
  DQMStore store(emptyps);	
  for (int arg = 2; arg < argc; ++arg)
    try
    {
      // Read in the file.
      store.open(argv[arg]);
    }
    catch (std::exception &e)
    {
      std::cerr << "*** FAILED TO READ FILE " << argv[arg] << ":\n"
		<< e.what() << std::endl;
      exit(1);
    }

  store.save(output);
  return 0;
}
