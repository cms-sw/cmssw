#ifndef ClientRoot_h
#define ClientRoot_h

#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <queue>

#include <SealBase/Callback.h>

#include <TTimeStamp.h>
#include "DQMServices/Core/src/ReceiverBase.h"

class TString;
class DQMMessage;

class ClientRoot : public ReceiverBase
{
  
 public:
  /// ctor #1: configure client with collector hostname & port it should connect to
  ClientRoot(std::string host, unsigned port, std::string client_name,
	     bool keepStaleSources = false, bool multipleConnections = true);
  /// ctor #2: client configuration for connection w/ multiple collectors
  ClientRoot(std::vector<std::string> hosts, unsigned port, 
	     std::string client_name, bool keepStaleSources = false);
  virtual ~ClientRoot();

  /// ============= Connection related methods ================

  /// attempt to connect to hostname at port #;
  /// Use isConnected to test if connection has been made
  void connect(std::string host, unsigned port);
  /// opposite action of connect method
  void disconnect(std::string host);
  /// disconnect from all (connected or not) collectors
  void disconnect();
  /// true if connection with server/collector was succesful
  bool isConnected(std::string host) const;
  /// true if at least one connection was succesful
  bool isConnected() const;
  /// true if all connections were succesful
  bool allConnected() const;

  
  /// receive monitoring/send subscription requests; return success flag
  bool update();
  /// return # of monitoring cycles received
  inline int getNumUpdates(void) const {return updates_;}
  /// upon a collector crash, the client will automatically attempt
  /// to reconnect with a time delay (secs); use method to set parameter
  inline void setReconnectDelay(unsigned delay_secs)
    {NodeBase::setReconnectDelay(delay_secs);}
  /// set maximum # of attempts to reconnect to server (upon connection problems)
  inline void setMaxAttempts2Reconnect(unsigned Nrecon_attempts)
  {maxAttempts2Reconnect = Nrecon_attempts;}
 
  static int ss; 	 // listening server for incoming connections/messages
 
  

 protected:

  struct ConAttempt_ {
    TTimeStamp when; /// time of last connection attempt
    unsigned attempts; /// # of (failed) attempts to retry
    ConAttempt_(){when = TTimeStamp(); attempts = 0;}
  };
  typedef struct ConAttempt_ ConAttempt;
  /// key: hostname of server; data: 
  typedef std::map<std::string, ConAttempt> HostConAttemptMap;
  /// collectors whose connection is pending;
  /// will keep checking on them till they are all connected!
  HostConAttemptMap collectorsPending;

  /// add socket to set of senders
  /// (i.e. set of sources if this is a collector, or set of collectors if this
  /// is a client)
  void addSender(std::string name);

  /// key: node name or hostname, value: socket fd
  typedef std::map<std::string, int> NameSocketMap;
  /// map containing hostnames for all servers
  /// (useful for looking up hostnames quickly in client)
  NameSocketMap serv_hostnames;

  /// map containing names for all nodes (sources + clients)
  /// (useful when need to access sockets by name)
  NameSocketMap all_names;

  typedef std::map<int , bool> SocketBoolMap; 
  /// map containing all sockets
  /// int is the file descriptor of the socket
  /// (useful for looking up if node is source or client quickly)
  SocketBoolMap isSource;

  /// key: socket fd, value: server (source or collector)
  typedef std::map<int, SenderData> SourceMap;
  ///map containing data for all sources (if this is a collector) or collectors 
  /// (if this is a client);
  /// add items with addSource; remove items with removeSocket
  SourceMap sources;

  /// key: hostname, value: node name
  /// use this data member to keep track of disconnected nodes;
  /// (useful when ReceiverBase::keepStaleSources_ = true and node re-connects)
  std::map<std::string, std::string> droppedNodes;

  /// true if connection with server was succesful
  bool isConnected(int s) const;
  /// disconnect from socket (if connected), release memory
  void disconnect(int s) const;

  /// set maximum time (msecs) to wait for monitoring updates before sending
  /// sending subscription requests 
  /// (default: <defTimeoutMsecs> msecs for clients, -1 for collectors)
  inline void setWait2Subscribe(int time_msecs) {timeout_ = time_msecs;}

  /// will set timeout_ for clients (collectors should override initial value)
  static const int defTimeoutMsecs = 1000;
  /// maximum # of socket connections
  /// do we really need this?
  static const unsigned MAX_CONN = 128;

  /// get active socket (private member "currs" setting)
  void selectSock();

  std::string getHostname(int s) const;
 
  /// come here when sender goes down; will try to reconnect (if applicable)
  void senderIsDown(bool graceful_exit, bool add2collPending);
  /// come here when (previously disconnected) sender has re-connected
  void senderHasRecovered();
  /// basic cleaning-up, applicable to both sources & clients
  void removeSocket_base(std::string name);
  /// remove source when disconnected;
  /// to be redefined by inheriting class
  virtual 
    void removeSocket(int s, bool graceful_exit, bool add2collPending);

  /// check all pending server connections; 
  /// if problems try to reconnect (if applicable)
  void checkPendingConnections();
  /// b4 calling receiver: set addresses for socket, message & monitoring structure
  void setReceiverPtrs();
  /// check if node has sent name & is identified as source; 
  virtual bool newNode(std::string & buffer);
  /// check if node has sent name & is identified as source; 
  /// add to source list if appropriate
  bool newSource(const std::string & buffer);
  /// here we attempt to send subscription requests to sources
  void doSourceLoop();
  /// attempt to receive monitoring, monitorable; 
  /// returns true if full monitoring set has been received
  bool receiveStuff(void);
  /// attempt to send monitoring, quality test results
  virtual void sendStuff(){}
  /// returns true if full monitoring set (or command) has been received
  virtual bool unpackData();
  /// # of disconnected sources
  unsigned closedSources;
  /// to be implemented by inheriting class
  virtual void summary(bool checkMemoryLeaks = false){}
  /// true if no upstream connection is made or pending
  bool noUpstreamConnections() const;
  /// will look for <hostname> in droppedNodes; return success;
  /// if found, modify sname to previous name if necessary
  bool checkDroppedNodes(std::string & sname, std::string hostname);
  /// insert "name" to all_names; if name already exists, 
  /// insert modified "name" --> "name_n" instead (n: integer); 
  /// Will also check list of dropped nodes
  void insertName(std::string & name);
  /// return # of XXX instances in ROOT memory (specified by name)
  int getROOTcount(const TString & name) const;
  /// printout of monitoring objects in memory
  void checkMemory();
  /// add socket to isSource
  void addSocket(int sock);
  /// come here to attempt connection with UPSTREAM server/collector at <host>
  void connect2Server(const std::string & host);
  ///
  bool actAsServer_;
  bool shouldResetMEs;
  bool shouldCallResetDiff;
  /// set this to true for nodes with no upstream connection
  /// (e.g. sources or clients in "standalone mode")
  inline void setShouldResetMEs(bool flag){shouldResetMEs = flag;}
  /// if you set this to false, you must call resetDiff yourself
  /// (e.g. at beginning of monitoring cycle for class MonitorUserInterface)
  inline void setShouldCallResetDiff(bool flag){shouldCallResetDiff = flag;}

  inline bool isServer() const {return actAsServer_;}
  DQMMessage *mess;  // active message
  int currs; // active socket descriptor
  std::ofstream file;

  struct callback_queue_ {
    LockMutex::Mutex mutex;
    /// queue with callback methods to be called in update method
    std::queue<seal::Callback> cmds;
  };
  typedef struct callback_queue_ callback_queue;

  callback_queue cb_queue;

  /// when false, TMonitor will not be instantiated
  bool multipleConnections_; 

  /// false if multiple connections with (>) collectors, or downstream connections
  bool multipleConnections() const {return multipleConnections_;}

  /// return server socket; used to handle ctrl-c termination of collector process
  static int getServerSocket (void);
 

 private:
  /// maximum time (msecs) to wait for monitoring updates before sending 
  /// subscription requests (setting socket timeout); use setWait2Subscribe to set;
  /// this should be ~O(sec) for clients, and <0 (ie. disabled) for collectors
  int timeout_;
  /// initialization done at startup
  void init(unsigned port, int timeout_msecs);
  /// set port #
  inline void setPort(unsigned port){NodeBase::port_ = port;}
  /// set host
  inline void setHost(const std::string & host){NodeBase::host_ = host;}
  /// come here when server socket notifies of new connection
  void newConnection(int s);
  /// true if <name_check> does not appear in all_names or droppedNodes
  bool unique_name(const std::string & name_check) const;

  /// # of monitoring cycles received
  int updates_;
  int monMax;                              //max fd in sockets
  bool selectCalled;
  fd_set rmask;
 
  unsigned maxAttempts2Reconnect;

  /// false by default; true only if client runs in standalone mode
  bool doNotBlockUpdate; 

  /// add call back; to be used for thread-unsafe operations
//  void addCallback(seal::Callback & action);
  /// run queue of callback methods
  void run_callbacks();

  /// get active socket
  /// to be used when having multiple connections
  void selectSockTMon();
  /// get active socket 
  /// to be used with single connection
  void selectSockSimple();


  friend class MonitorUIRoot;
  friend class MonitorUserInterface;

};

#endif
