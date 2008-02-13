
#include "DQMServices/Core/interface/MonitorUIRoot.h"

MonitorUIRoot::MonitorUIRoot() : MonitorUserInterface() {}

// destructor
MonitorUIRoot::~MonitorUIRoot(void)
{
}


MonitorUIRoot::MonitorUIRoot(const std::string &hostname, int port_no,
			     const std::string &client_name, 
			     int reconnect_delay_secs, bool actAsServer) :
		MonitorUserInterface(hostname,port_no,client_name,
		             reconnect_delay_secs,actAsServer) { }
