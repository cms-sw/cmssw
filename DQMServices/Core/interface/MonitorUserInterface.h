#ifndef MonitorUserInterface_h
#define MonitorUserInterface_h

class MonitorElement;

#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/QCriterion.h"

#include "DQMServices/Core/src/ClientRoot.h"
#include "DQMServices/Core/src/ClientServerRoot.h"
#include "DQMServices/Core/interface/MonitorElementRootT.h"

#include <SealBase/Callback.h>

#include <list>
#include <map>

using namespace dqm::me_util;
using std::cout; using std::cerr; using std::endl;
using std::string; using std::vector;

class MonitorElementRootFolder;

class MonitorUserInterface : public StringUtil
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
  
  MonitorUserInterface();
 
  /// Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
  /// using <client_name>;
  MonitorUserInterface(std::string hostname, int port_no, std::string client_name,
		/// use delay < 0 for no reconnection attempts
		int reconnect_delay_secs = 5,
		/// if flag=true, client will accept downstream connections
		bool actAsServer = false);

  /// Connect with monitoring server (DQM Collector) with a list of hostnames at 
  /// <port_no> using <client_name>; 
  MonitorUserInterface(std::vector<std::string> hostnames, int port_no, 
		std::string client_name, int reconnect_delay_secs = 5,
		/// if flag=true, client will accept downstream connections
		bool actAsServer = false);

  virtual ~MonitorUserInterface();

  /// get pointer to back-end interface
  DaqMonitorBEInterface * getBEInterface(void){return bei;}

  /** this is the "main" loop where we receive monitoring or
      send subscription requests;
      if client acts as server, method runQTests is also sending monitoring & 
      test results to clients downstream;
      returns success flag */
  bool update(void);
  bool doMonitoring(void);

  // needed for IGUANA and/or DQMServer/bin/collector.cpp
  bool isConnected(void) ;
  void disconnect(void) { 
                  if (!standaloneMode_) myc->disconnect();}
  void connect(std::string host, unsigned port) { 
                  if (!standaloneMode_) myc->connect(host,port);}
  /// set(unset) subscription if add=true(false)
  void finishSubscription(const std::vector<std::string> & monit, bool add);


private:

  /// use to get hold of structure with monitoring elements that class owns
  DaqMonitorBEInterface *bei;
  /// client pointer
  ClientRoot * myc;
  /// to be called by ctors
  void init();

  /// when in "standalone mode", there are no upstream connections
  /// (but there may be downstream clients...)
  bool needUpstreamConnections() const {return !standaloneMode_;}
  // 
  bool standaloneMode_;


  // *****************  NOTE ******************************************
  /// instead of calling "update", users can use the following two methods:
  /// retrieval of monitoring, sending of subscription requests/cancellations,
  /// calculation of "collate"-type Monitoring Elements;

  /// allow downstream clients to connect to this client
  /// (to be used only with no-arg ctor; use boolean switch for other ctors)
  void actAsServer(int port_no, std::string client_name);
 
  /// true if client accepts downstream connections/requests
  inline bool isServer() const {
        if(myc) return myc->isServer(); return false; }

  /// return # of monitoring cycles received
  int getNumUpdates(void) const;
  
  /// set maximum # of attempts to reconnect to server (upon connection problems)
  void setMaxAttempts2Reconnect(unsigned Nrecon_attempts) {
        if(myc) myc->setMaxAttempts2Reconnect(Nrecon_attempts); }

  // ---------------- Subscriptions -----------------------------
  
  /// subscription request; format: 
  /// (a) exact pathname with ME name (e.g. A/B/C/histo) ==> FAST
  void subscribe(std::string subsc_request);
  void subscribeNew(std::string subsc_request);
  void unsubscribe(std::string subsc_request);
  void subscribe_base(const std::string & subsc_request, bool add,
		      const dqm::me_util::rootDir & Dir);
  
  /// (un)subscription request for directory contents ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C);
  /// use flag to specify whether subfolders (& their contents) should be included;
  /// use add=true(false) to (un)subscribe
  //  void subscribeDir(std::string & subsc_request, bool useSubFolders,
  //		    unsigned int myTag, bool add);
  /// same as above for MonitorElementRootFolder
  void subscribeDir(MonitorElementRootFolder * folder, bool useSubFolders,
			    unsigned int myTag, bool add);

  
  /// # of monitoring packages received;
  int numUpdates_;
  
  /// like subscribe_base in base class, for one folder only
  void subscribe_base(const std::string & subsc_request, bool add,
		      std::vector<std::string> & requests, 
		      const MonitorElementRootFolder * folder);

  /// add <pathname:> or <pathname::myTag> (if myTag != 0) to requests
  void addFolderSubsc(MonitorElementRootFolder * folder, unsigned int myTag, 
		      std::vector<std::string> & requests) const;

  // needed by VisDQMMonitorService
  void setReconnectDelay(int delay){if(myc)myc->setReconnectDelay(delay);} 


  friend class VisDQMMonitorService ;
  friend class DQMBaseClient ;
  friend class SiPixelHistoricInfoClient; //needs subscribe and getNumUpdates
  friend class SiStripHistoricInfoClient; //needs subscribe and getNumUpdates
  

};

#endif
