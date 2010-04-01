#undef WITHOUT_CMS_FRAMEWORK
#define WITHOUT_CMS_FRAMEWORK 1
#include "DQMServices/Core/interface/DQMNet.h"
#include "DQMServices/Core/src/DQMNet.cc"
#include "DQMServices/Core/src/DQMError.cc"
  /* NB: Avoids picking up the entire DQMServices/Core
         library, and in particular avoids ROOT. */
#include "classlib/utils/DebugAids.h"
#include "classlib/utils/Signal.h"
#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <cstdlib>

static const int FATAL_OPTS = (lat::Signal::FATAL_DEFAULT
			       & ~(lat::Signal::FATAL_ON_INT
				   | lat::Signal::FATAL_ON_QUIT
				   | lat::Signal::FATAL_DUMP_CORE));

volatile sig_atomic_t s_stop = 0;
static void
interrupt (int sig) 
{
  s_stop = 1;
}

class DQMCollector : public DQMBasicNet
{
public:
  bool shouldStop(void)
    {
      return s_stop != 0;
    }
  
  DQMCollector(char *appname, int port, bool debugging)
    : DQMBasicNet (appname)
    {
      // Establish the server side.
      debug(debugging);
      startLocalServer(port);
    }
};

//////////////////////////////////////////////////////////////////////
// Always abort on assertion failure.
static char
onAssertFail (const char *message)
{
  std::cout.flush();
  fflush(stdout);
  std::cerr.flush();
  fflush(stderr);
  std::cerr << message << "ABORTING\n";
  return 'a';
}

//////////////////////////////////////////////////////////////////////
// Run the main program.
int main (int argc, char **argv)
{
  lat::DebugAids::failHook(&onAssertFail);
  lat::Signal::handleFatal(argv[0], IOFD_INVALID, 0, 0, FATAL_OPTS);
  lat::Signal::handle(SIGINT, (lat::Signal::HandlerType) &interrupt);
  lat::Signal::ignore(SIGPIPE);

  // Check and process arguments.
  int port = 9090;
  bool debug = false;
  bool bad = false;
  for (int i = 1; i < argc; ++i)
    if (i < argc-1 && ! strcmp(argv[i], "--listen"))
      port = atoi(argv[++i]);
    else if (! strcmp(argv[i], "--debug"))
      debug = true;
    else if (! strcmp(argv[i], "--no-debug"))
      debug = false;
    else
    {
      bad = true;
      break;
    }

  if (bad || ! port)
  {
    std::cerr << "Usage: " << argv[0] << " --listen PORT [--[no-]debug]\n";
    return 1;
  }

  // Start serving.
  DQMCollector server (argv[0], port, debug);
  server.run();
  exit(0);
}
