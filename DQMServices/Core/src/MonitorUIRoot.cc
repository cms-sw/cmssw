
#include "DQMServices/Core/interface/MonitorUIRoot.h"

MonitorUIRoot::MonitorUIRoot() : MonitorUserInterface() {}

// destructor
MonitorUIRoot::~MonitorUIRoot(void)
{
}


MonitorUIRoot::MonitorUIRoot(vector<string> hostnames, int port_no, 
			     string client_name, int reconnect_delay_secs,
			     bool actAsServer) : 
		MonitorUserInterface (hostnames,port_no,client_name,
		             reconnect_delay_secs,actAsServer) { }

MonitorUIRoot::MonitorUIRoot(string hostname, int port_no, string client_name, 
			     int reconnect_delay_secs, bool actAsServer) :
		MonitorUserInterface(hostname,port_no,client_name,
		             reconnect_delay_secs,actAsServer) { }
