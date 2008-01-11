#include "DQMServices/Core/interface/CollectorRoot.h"

using std::string;
using std::cout;
using std::endl;

CollectorRoot::CollectorRoot(string name, int listenport, bool keepStaleSources)
  : ClientServerRoot("", listenport, name, keepStaleSources, false),
    inputAvail_(false), listenport_(listenport)
{
  // see comment in ClientRoot::shouldResetMEs
  setShouldResetMEs(false); 
  // see comment in ClientRoot::setShouldCallResetDiff
  setShouldCallResetDiff(true);

  setWait2Subscribe(-1); // disable time-out
}

CollectorRoot::~CollectorRoot(void)
{

}

// "main" loop:
void CollectorRoot::run()
{  
  cout << " DQM Server (aka Collector) started at port " << listenport_ 
       << endl;
  bool stay_in_loop = true;

  while (stay_in_loop) 
    {
      // infinite loop
      stay_in_loop = run_once();
    }
}

// the REAL main loop
bool CollectorRoot::run_once(void)
{
  if(update())// if we receive monitoring/monitorable
    sendStuff(); // send it downstream...
  return true;
}
