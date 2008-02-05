#ifndef MonitorData_h
#define MonitorData_h

#include <string>
#include <iostream>

namespace dqm
{
  namespace monitor_data
  {
    /// messages sent between nodes identifying
    static const std::string nameClientPrefix = "ClientName;";
    static const std::string nameSourcePrefix = "SourceName;";
    static const std::string listAddPrefix = "ListAdd;";
    static const std::string listRemPrefix = "ListRem;";
    static const std::string subscAddPrefix = "SubscAdd;";
    static const std::string subscRemPrefix = "SubscRem;";
    static const std::string objectsPrefix = "Objects;";

    static const std::string tagAddPrefix = "TagAdd;";
    static const std::string tagRemPrefix = "TagRem;";

    static const std::string Quitting = "Quit;";
    

    static const std::string DummyNodeName = "DummyNode";

    /// done sending monitoring objects for current cycle
    static const unsigned kMESS_DONE_MONIT_CYCLE = 10010; 
    /// done sending monitorable
    static const unsigned kMESS_DONE_MONITORABLE = 10011;
    /// done sending subscription
    static const unsigned kMESS_DONE_SUBSCRIPTION = 10012; 
    
    /// use for intercommunication among classes: > 0 for messages, <0 for errors
    static const int DO_STATISTICS = 20000; /// show statistics
    
    /// default port for communication with collector
    static const unsigned COLLECTOR_PORT = 9090;
    /// default monitoring period  (in msecs);
    /// i.e. how often monitoring elements are shipped over
    static const unsigned SEND_PERIOD = 1000;
    /// default name of DQM source
    static const std::string SOURCE_NAME = "DQMSource";
    /// upon a connection problem, the source will automatically attempt
    /// to reconnect with a time delay (secs); this is the default value
    static const unsigned RECONNECT_DELAY = 5;
    /// default value for maximum # of reconnection attempts
    static const unsigned MAX_RECON = 10;

  }
}
#endif // MonitorData_h
