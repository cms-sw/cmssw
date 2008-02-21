#ifndef DQMSERVICES_CORE_DQM_CONNECTOR_H
# define DQMSERVICES_CORE_DQM_CONNECTOR_H

# include <string>

class DQMNet;
class DQMStore;
class DQMOldReceiver
{

public:

  /** Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
     using <client_name>; if flag=true, client will accept downstream connections
     MonitorUserInterface(std::string hostname,int port_no,std::string client_name,

     int reconnect_delay_secs = 5, bool actAsServer = false); 
  */

  /** Connect with monitoring server (DQM Collector) with a list of hostnames at 
     <port_no> using <client_name>;   
     if flag=true, client will accept downstream connections
     MonitorUserInterface::MonitorUserInterface(std::vector<std::string> hostnames, int port_no, 
     std::string client_name, int reconnect_delay_secs=5, bool actAsServer=false); 
  */ 

  /** Use the default constructor for running in standalone mode (ie. without
     sources or collectors); if flag=true, client will accept downstream connections
  */
  
  DQMOldReceiver(void);
 
  /// Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
  /// using <client_name>;
  DQMOldReceiver(const std::string &hostname, int port,
	       const std::string &client_name,
	       int unusedReconnectDelaySecs = -1,
	       bool unusedActAsServer = false);

  ~DQMOldReceiver(void);

  /// get pointer to back-end interface
  DQMStore *getStore(void)
    { return store_; }
  DQMStore *getBEInterface(void)
    { return getStore(); }

  /** this is the "main" loop where we receive monitoring or
      send subscription requests;
      if client acts as server, method runQTests is also sending monitoring & 
      test results to clients downstream;
      returns success flag */
  bool update(void);
  bool doMonitoring(void);
  int getNumUpdates(void) const { return numUpdates_; }

private:
  /// use to get hold of structure with monitoring elements that class owns
  DQMStore *store_;
  /// client pointer
  DQMNet *net_;
  int numUpdates_;
};

#endif // DQMSERVICES_CORE_DQM_CONNECTOR_H
