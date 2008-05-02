#include "DQMServices/Core/interface/DQMNet.h"
#include "classlib/utils/DebugAids.h"
#include <iostream>
#include <signal.h>
#include <unistd.h>
#include "TROOT.h"

sig_atomic_t s_stop = 0;
static void interrupt (int sig) 
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

  // Initialise ROOT.
  ROOT::GetROOT();
  signal(SIGINT, &interrupt);

  // Start serving.
  DQMCollector server (argv[0], port, debug);
  server.run();
  exit(0);
}
