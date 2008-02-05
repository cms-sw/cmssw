#ifndef MonitorDaemon_H
#define MonitorDaemon_H

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/src/RootMonitorThread.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

/** class responsible for shipping monitoring information and registering
    monitoring requests in secondary thread */
class MonitorDaemon 
{
public:
  MonitorDaemon(const edm::ParameterSet &pset);

  ~MonitorDaemon(void);

  RootMonitorThread *rmt(std::string, int port_no, int period_microsecs, 
				std::string source_name, int reconnect_delay);
  
  void reParseConfig(const edm::ParameterSet &pset)
  {
  }

  /// set maximum # of attempts to reconnect to server (upon connection problems)
  inline void setMaxAttempts2Reconnect(unsigned Nrecon_attempts)
    {if(od)od->setMaxAttempts2Reconnect(Nrecon_attempts);}
  
private:
  
  /// # of events to run daemon
  unsigned maxEvents_;

  friend class DQMShipMonitoring;

  static RootMonitorThread *od;
  /// host name of DQM collector
  std::string destination_address;
  /// port for communication w/ DQM collector 
  /// (default: dqm::monitor_data::COLLECTOR_PORT)
  int send_port;
  /// monitoring period (in msecs); default: dqm::monitor_data::SEND_PERIOD
  /// i.e. how often monitoring elements are shipped over
  int primary_delay;
  /// name of source (default: default: dqm::monitor_data::SOURCE_NAME)
  std::string name_as_source;
  /// whether DQM daemon should be started; 
  /// if false users must instantiate themselves with class RootMonitorThread
  bool auto_instantiating;
  /// if >= 0, upon a connection problem, the source will automatically attempt
  /// to reconnect with a time delay (secs)
  int reconnect_delay;
  /// maximum # of reconnection attempts upon connection problems
  /// (default value: NodeBase::MAX_RECON)
  unsigned maxAttempts2Reconnect;
 
};

#endif
