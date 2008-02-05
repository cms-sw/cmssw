#include "DQMServices/Core/src/AuxCollectorRoot.h"

using std::string;
using std::cout;
using std::endl;
using std::vector;
using namespace dqm::me_util;

AuxCollectorRoot::AuxCollectorRoot(string host, string name, int listenport)
  : CollectorRoot(name, listenport), listenport_(listenport)
{
  connect2Server(host);
}

AuxCollectorRoot::AuxCollectorRoot(vector<string> hosts, string name, 
				   int listenport)
  : CollectorRoot(name, listenport), listenport_(listenport)
{
  for(cvIt it = hosts.begin(); it != hosts.end(); ++it)
    connect2Server(*it);
}

AuxCollectorRoot::~AuxCollectorRoot(void)
{

}

// the REAL main loop
bool AuxCollectorRoot::run_once(void)
{
  if(noUpstreamConnections())
    {
      cout << " *** No connections with collectors are possible..." << endl;
      return false;
    }

  if(update())// if we receive monitoring/monitorable
    sendStuff(); // send it downstream...

  return true;
}
