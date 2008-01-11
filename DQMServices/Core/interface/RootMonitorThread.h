#ifndef RootMonitorThread_h
#define RootMonitorThread_h

#include "Utilities/Threads/interface/Thread.h"
#include "Utilities/General/interface/MutexUtils.h"

#include "DQMServices/Core/interface/SenderBase.h"

#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include <string>

class DQMMessage;

class RootMonitorThread : public Thread, public SenderBase
{

public:
  

  /// del: time delay in between shipments (in microsecs)
  /// recon_del: see NodeBase::reconnectDelay_
  RootMonitorThread(std::string host, unsigned port, unsigned del, 
		    std::string name, edm::ServiceToken iToken,
		    int recon_del); 
  ~RootMonitorThread(void);
  /// true if connection was succesful
  bool isConnected(void) const;
  /// attempt to connect (to be used if ctor failed to connect)
  void connect(std::string host, unsigned port);
  /// close connection
  void closeConnection(void);
  /// upon a collector crash, the source will automatically attempt
  /// to reconnect with a time delay (secs); use method to set parameter;
  /// use delay < 0 for no reconnection attempts
  inline void setReconnectDelay(int delay_secs)
  {NodeBase::setReconnectDelay(delay_secs);}
  /// set maximum # of attempts to reconnect to server (upon connection problems)
  inline void setMaxAttempts2Reconnect(unsigned Nrecon_attempts)
  {maxAttempts2Reconnect = Nrecon_attempts;}

  /// infinite monitoring loop
  void run(void); 
  /// lock
  void pause(void);
  /// unlock
  void release(void);
  void terminate(void);
  void abort(void);
  
private:
  
  SimpleLockMutex::Mutex genLock;
  boost::mutex::scoped_lock *lock;
  
  // read documentation of SenderBase::doneSending
  static const bool resetMEs; static const bool callResetDiff;

  // if true (default), will ship monitoring objects in 2ndary thread;
  // can be turned off by including DQMShipMonitoring module
  bool shipMonitoringIn2ndThread;

  friend class DQMShipMonitoring;

  int sock;                 // receiver socket
  int maxSock;              // maximum fd (needed when calling select)
  fd_set rmask;             // mask of active sockets
  DQMMessage *mess;         // main message

  std::string recv_name; 
  bool terminate_;
  /// used to setup services in new thread (see ctor & method run)
  edm::ServiceToken token_;
  /// holds data for receiving end
  ReceiverData receiver;
  /// b4 calling sender: set addresses for socket and monitoring structure
  void setSenderPtrs(void);
  /// maximum # of consecutive failures in sending allowed
  int s_fail_consec_max;
  /// false if collector is not responsive
  bool isCollectorAlive(void);
  /// come here to connect once host & port have been determined
  void connect(void);
  /// wait N secs, then attempt to connect
  void wait_and_connect(unsigned N);
  /// come here when collector has died; try to resurrect connection
  void recoverCollector(void);
  /// return connection status
  bool checkConnection(void);
  /// come here when collector has died; recover connection, or return
  void collectorIsDead(void);
  /// sending monitorable & monitoring
  void sendStuff(void);
  /// receive (un)subscription requests (wait up to <wait_msecs> msecs)
  void receiveStuff(int wait_msecs);
  /// send monitoring only (to be called by DQMShipMonitoring)
  void sendMonitoringOnly();

  ///
  unsigned maxAttempts2Reconnect;
  /// wait <reconnectDelay_> secs, then attempt to connect, 
  /// up to maxAttempts2Reconnect times
  void wait_and_connect();

};
#endif // RootMonitorThread_h
