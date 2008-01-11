#ifndef MonitorUIRoot_h
#define MonitorUIRoot_h

#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/src/ClientServerRoot.h"

#include <string>
#include <vector>

class MonitorElement;
class ClientRoot;
class MonitorElementRootFolder;

class MonitorUIRoot: public MonitorUserInterface
{

 public:
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
  /// Use the default constructor for running in standalone mode 
  /// (ie. without sources or collectors);
  MonitorUIRoot();
  ///
  virtual ~MonitorUIRoot();

  /// allow downstream clients to connect to this client
  /// (to be used only with no-arg ctor; use boolean switch for other ctors)
  void actAsServer(int port_no, std::string client_name);
 
  /// true if client accepts downstream connections/requests
  inline bool isServer() const
  {
    if(myc)
      return myc->isServer();
    else
      return false;
  }

  /// --------------------------- Getters -----------------------------
  /// get ME from full pathname (e.g. "my/long/dir/my_histo")
  MonitorElement * get(const std::string & fullpath) const;
  /// return # of monitoring cycles received
  int getNumUpdates(void) const;
  /// set maximum # of attempts to reconnect to server (upon connection problems)
  void setMaxAttempts2Reconnect(unsigned Nrecon_attempts)
  {if(myc)myc->setMaxAttempts2Reconnect(Nrecon_attempts);}

  /// retrieval of monitoring, sending of subscription requests/cancellations,
  /// calculation of "collate"-type Monitoring Elements;
  /// returns success flag
  bool doMonitoring(void);
  /// Run quality tests on all MonitorElements that have been updated (or added)
  /// since last monitoring cycle;
  /// send monitoring, results downstream if applicable
  void runQTests(void);

  /// ---------------- Miscellaneous -----------------------------
  
  /// attempt to connect (to be used if ctor failed to connect)
  void connect(std::string host, unsigned port)
  {if(needUpstreamConnections())myc->connect(host, port);}
  /// opposite action of that of connect method
  void disconnect(void){if(needUpstreamConnections())myc->disconnect();}
  /// true if connection was succesful
  virtual bool isConnected(void) const
  {
    if (!needUpstreamConnections())
      return false;

    return myc->isConnected();
  }
  /// set reconnect delay parameter (in secs);
  /// use delay < 0 for no reconnection attempts
  void setReconnectDelay(int delay){if(myc)myc->setReconnectDelay(delay);}


  /// add call back; to be used for thread-unsafe operations
  void addCallback(seal::Callback & action);

 private:
  /// to be called by ctors
  void init();
  /// client pointer
  ClientRoot * myc;

  /// like subscribe_base in base class, for one folder only
  void subscribe_base(const std::string & subsc_request, bool add,
		      std::vector<std::string> & requests, 
		      const MonitorElementRootFolder * folder);

  /// (un)subscription request for directory contents ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C);
  /// use flag to specify whether subfolders (& their contents) should be included;
  /// use add=true(false) to (un)subscribe
  void subscribeDir(std::string & subsc_request, bool useSubFolders,
		    unsigned int myTag, bool add);
  /// same as above for MonitorElementRootFolder
  void subscribeDir(MonitorElementRootFolder * folder, bool useSubFolders,
			    unsigned int myTag, bool add);

  /// add <pathname:> or <pathname::myTag> (if myTag != 0) to requests
  void addFolderSubsc(MonitorElementRootFolder * folder, unsigned int myTag, 
		      std::vector<std::string> & requests) const;

  /// true if name appear as child of directory (even if it is null MonitorElement)
  bool objectDefined(const std::string & name, const std::string & pathname) 
    const;
};


#endif
