#ifndef MonitorUserInterface_h
#define MonitorUserInterface_h

class MonitorElement;
class CollateMonitorElement;

#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/QCriterion.h"

#include <SealBase/Callback.h>

#include <list>
#include <map>

class MonitorElementRootFolder;

class MonitorUserInterface : public StringUtil
{

 protected:
  /** Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
     using <client_name>; if flag=true, client will accept downstream connections
     MonitorUserInterface(std::string hostname,int port_no,std::string client_name,

     int reconnect_delay_secs = 5, bool actAsServer = false); */

  /** Connect with monitoring server (DQM Collector) with a list of hostnames at 
     <port_no> using <client_name>;   
     if flag=true, client will accept downstream connections
     MonitorUIRoot::MonitorUIRoot(std::vector<std::string> hostnames, int port_no, 
     std::string client_name, int reconnect_delay_secs=5, bool actAsServer=false); 
  */ 

  /** Use the default constructor for running in standalone mode (ie. without
     sources or collectors); if flag=true, client will accept downstream connections
  */
  MonitorUserInterface();
 
  /// when in "standalone mode", there are no upstream connections
  /// (but there may be downstream clients...)
  bool needUpstreamConnections() const {return !standaloneMode_;}
  // 
  bool standaloneMode_;


 public:

  virtual ~MonitorUserInterface();

  /// get pointer to back-end interface

  DaqMonitorBEInterface * getBEInterface(void){return bei;}
  

  // ------------------ Updates ------------------------------------
  /// add call back; to be used for thread-unsafe operations
  virtual void addCallback(seal::Callback & action) = 0;

  /** this is the "main" loop where we receive monitoring or
      send subscription requests;
      if client acts as server, method runQTests is also sending monitoring & 
      test results to clients downstream;
      returns success flag */
  bool update(void);

  // *****************  NOTE ******************************************
  /** instead of calling "update", users can use the following two methods:
      1. Retrieval of monitoring, sending of subscription requests/cancellations,
      calculation of "collate"-type Monitoring Elements;
      returns success flag */
  virtual bool doMonitoring(void) = 0;

  /// allow downstream clients to connect to this client
  /// (to be used only with no-arg ctor; use boolean switch for other ctors)
  virtual void actAsServer(int port_no, std::string client_name) = 0;
  /// true if client accepts downstream connections/requests
  virtual bool isServer() const = 0;

  /// return # of monitoring cycles received
  virtual int getNumUpdates(void) const = 0;
  /// set maximum # of attempts to reconnect to server (upon connection problems)
  virtual void setMaxAttempts2Reconnect(unsigned Nrecon_attempts) = 0;


  // ---------------- Subscriptions -----------------------------
  
  /// subscription request; format: 
  /// (a) exact pathname with ME name (e.g. A/B/C/histo) ==> FAST
  /// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*) ==> SLOW
  void subscribe(std::string subsc_request);
  /// subscription request for directory contents ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C)
  /// use flag to specify whether subfolders (& their contents) should be included;
  /// Users are encourage to use this method instead of previous one w/ wildcards
  void subscribe(std::string subsc_request, bool useSubFolders);
  
  /// same as above for tagged MonitorElements
  void subscribe(std::string subsc_request, unsigned int tag);
  void subscribe(std::string subsc_request, bool useSubFolders, 
		 unsigned int tag);
  /// subscription request for all MEs with given tag ==> FAST
  void subscribe(unsigned int tag);

  /// similar to above subscription methods; 
  /// use only additions to monitorable in last cycle
  void subscribeNew(std::string subsc_request);
  void subscribeNew(std::string subsc_request, bool useSubFolders); 
  void subscribeNew(std::string subsc_request, unsigned int tag);
  void subscribeNew(std::string subsc_request, bool useSubFolders,
		    unsigned int tag);
  void subscribeNew(unsigned int tag);

  /// unsubscription requests; format is same with subscription requests 
  void unsubscribe(std::string subsc_request);
  void unsubscribe(std::string subsc_request, bool useSubFolders);
  void unsubscribe(std::string subsc_request, unsigned int tag);
  void unsubscribe(std::string subsc_request, bool useSubFolders,
		   unsigned int tag);
  ///
  void unsubscribe(unsigned int tag);

  // ---------------- Miscellaneous -----------------------------
  
  /// attempt to connect (to be used if ctor failed to connect)
  virtual void connect(std::string host, unsigned port) = 0;
  /// opposite action of that of connect method
  virtual void disconnect(void) = 0; 
  /// true if connection was succesful
  virtual bool isConnected(void) const = 0;
  /// set reconnect delay parameter (in secs);
  /// use delay < 0 for no reconnection attempts
  virtual void setReconnectDelay(int delay) = 0;

  // --------------------- Collation of MEs --------------------------------
  /// collate 1D histograms, store in <pathname>
  virtual CollateMonitorElement * collate1D(const std::string name, 
					    const std::string title, 
					    const std::string pathname)=0;
  /// collate 2D histograms, store in <pathname>
  virtual CollateMonitorElement * collate2D(const std::string name, 
					    const std::string title, 
					    const std::string pathname)=0;
  /// collate 3D histograms, store in <pathname>
  virtual CollateMonitorElement * collate3D(const std::string name, 
					    const std::string title, 
					    const std::string pathname)=0;
  /// collate profiles, store in <pathname>;
  virtual CollateMonitorElement* collateProf(const std::string name, 
					     const std::string title, 
					     const std::string pathname)=0;
  
  /// collate profiles, store in <pathname>;
  virtual CollateMonitorElement* collateProf2D(const std::string name, 
					       const std::string title, 
					       const std::string pathname)
    = 0;
  
  /// add <search_string> to summary ME; 
  /// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
  /// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*): SLOW
  /// this action applies to all MEs already available or future ones
  void add(CollateMonitorElement * cme, std::string search_string) const;
  /// same as above for tagged MEs
  void add(CollateMonitorElement * cme, unsigned int tag, 
	   std::string search_string) const;
  /// add directory contents to summary ME ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C);
  /// use flag to specify whether subfolders (& their contents) should be included;
  /// this action applies to all MEs already available or future ones
  void add(CollateMonitorElement* cme, std::string pathname, bool useSubfolds)
    const;
  /// same as above for tagged MEs
  void add(CollateMonitorElement* cme,unsigned int tag,std::string pathname,
	   bool useSubfolds) const;
  /// add tagged MEs to summary ME ==> FAST
  /// this action applies to all MEs already available or future ones
  void add(CollateMonitorElement * cme, unsigned int tag) const;
  /// do calculations for all collate MEs; come here at end of monitoring cycle)
  void doSummary(void);

  ///

  // -------------------- Quality tests on MonitorElements ------------------

 protected:

  /// to be used by methods subscribe (add=true) and unsubscribe (add=false)
  /// <subsc_request> format: (a) exact pathname (e.g. A/B/C/histo)
  /// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
  void subscribe_base(const std::string & subsc_request, bool add,
		      const dqm::me_util::rootDir & Dir);
  
  /// (un)subscription request for directory contents ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C);
  /// use flag to specify whether subfolders (& their contents) should be included;
  /// use add=true(false) to (un)subscribe
  virtual void subscribeDir(std::string & subsc_request, 
			    bool useSubFolders,
			    unsigned int myTag, bool add) = 0;

  /// same as above for MonitorElementRootFolder
  virtual void subscribeDir(MonitorElementRootFolder * folder, 
			    bool useSubFolders,
			    unsigned int myTag, bool add) = 0;

  /// look for MEs matching subsc_request with <tag> in <path>
  void subscribeNew(const std::string & subsc_request, unsigned int tag,
		    std::vector<std::string> & requests,
		    const dqm::me_util::cdirt_it & path);

  /// get all MEs with <tag> in <path>
  void subscribeNew(unsigned int tag, std::vector<std::string> & requests,
		    const dqm::me_util::cdirt_it & path);

  /// set(unset) subscription if add=true(false)
  void finishSubscription(const std::vector<std::string> & monit, bool add);
  
  /// collector name (e.g. <my_collector> or <collector1/collector2>
  std::string collector_name_;

  /// new MEs have been added; check if need to update collate-MEs
  void checkAddedContents(void);
  
 /// use to get hold of structure with monitoring elements that class owns
  DaqMonitorBEInterface *bei;
  /// # of monitoring packages received;
  int numUpdates_;
  
 private:
  /// like subscribe_base above, for one folder only
  virtual void subscribe_base
    (const std::string & subsc_request, bool add, 
     std::vector<std::string> & requests, 
     const MonitorElementRootFolder * folder) = 0;
  
};

#endif
