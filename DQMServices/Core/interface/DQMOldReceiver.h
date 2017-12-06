#ifndef DQMSERVICES_CORE_DQM_CONNECTOR_H
# define DQMSERVICES_CORE_DQM_CONNECTOR_H

# if __GNUC__ && ! defined DQM_DEPRECATED
#  define DQM_DEPRECATED __attribute__((deprecated))
# endif

# include <string>

class DQMStore;
class DQMOldReceiver
{

public:

  /** Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
     using <client_name>; if flag=true, client will accept downstream connections
     DQMOldReceiver(std::string hostname,int port_no,std::string client_name,

     int reconnect_delay_secs = 5, bool actAsServer = false); 
  */

  /** Connect with monitoring server (DQM Collector) with a list of hostnames at 
     <port_no> using <client_name>;   
     if flag=true, client will accept downstream connections
     DQMOldReceiver::DQMOldReceiver(std::vector<std::string> hostnames, int port_no, 
     std::string client_name, int reconnect_delay_secs=5, bool actAsServer=false); 
  */ 

  /** Use the default constructor for running in standalone mode (ie. without
     sources or collectors); if flag=true, client will accept downstream connections
  */
  
  DQMOldReceiver() DQM_DEPRECATED;
 
  /// Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
  /// using <client_name>;
  DQMOldReceiver(const std::string &hostname, int port,
	       const std::string &client_name,
	       int unusedReconnectDelaySecs = -1,
	       bool unusedActAsServer = false) DQM_DEPRECATED;

  ~DQMOldReceiver() DQM_DEPRECATED;

  /// get pointer to back-end interface
  DQMStore *getStore() DQM_DEPRECATED
    { return store_; }
  DQMStore *getBEInterface() DQM_DEPRECATED
    { return store_; }

  /** this is the "main" loop where we receive monitoring or
      send subscription requests;
      if client acts as server, method runQTests is also sending monitoring & 
      test results to clients downstream;
      returns success flag */
  bool update() DQM_DEPRECATED;
  bool doMonitoring() DQM_DEPRECATED;
  int getNumUpdates() const DQM_DEPRECATED { return 0; }

private:
  /// use to get hold of structure with monitoring elements that class owns
  DQMStore *store_;
} DQM_DEPRECATED;

#endif // DQMSERVICES_CORE_DQM_CONNECTOR_H
