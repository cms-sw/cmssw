#ifndef MonitorUIRoot_h
#define MonitorUIRoot_h

#include "DQMServices/Core/interface/MonitorUserInterface.h"

class MonitorUIRoot: public MonitorUserInterface
{

  public:
  /// Use the default constructor for running in standalone mode 
  /// (ie. without sources or collectors);
  MonitorUIRoot();
  ///
  virtual ~MonitorUIRoot();

  /// Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
  /// using <client_name>;
  MonitorUIRoot(std::string hostname, int port_no, std::string client_name,
		/// use delay < 0 for no reconnection attempts
		int reconnect_delay_secs = 5,
		/// if flag=true, client will accept downstream connections
		bool actAsServer = false);
  /// Connect with monitoring server (DQM Collector) with a list of hostnames at 
  /// <port_no> using <client_name>; 
  MonitorUIRoot(std::vector<std::string> hostnames, int port_no, 
		std::string client_name, int reconnect_delay_secs = 5,
		/// if flag=true, client will accept downstream connections
		bool actAsServer = false);

};


#endif
