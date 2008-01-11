#ifndef _SENDER_BASE_H_
#define _SENDER_BASE_H_

#include "DQMServices/Core/interface/NodeBase.h"
#include "DQMServices/Core/interface/MonitorData.h"

#include <string>
#include <vector>
#include <list>

class DQMMessage;
class MonitorElementRootFolder;
#include "DQMServices/Core/interface/DQMRootBuffer.h"

/** base class for clients that send (any kind of) monitoring information;the class 
   organizes the objects it owns through the NodeBase::bei (DaqMonitorBEInterface) 
   interface; (pointer receiver_ is to be set by inheriting class)
*/
class SenderBase : public NodeBase
{
 public:
  /// del: time delay in between shipments (in microsecs)
  SenderBase(std::string name, unsigned del, bool pureProducer = false);
  virtual ~SenderBase(void);
  
 private:
  /// flag for updated subscription info 
  /// (Sender needs to read subscription from Receiver)
  bool need2readSubscription;
  /// add objects to receiver's subscription list; 
  /// "name" format: see DirFormat definition;
  /// if new_request = true, send request to higher-level class if needed;
  /// return success flag
  bool addSubscription(const std::string & name, bool new_request = true);
  /// get monitorable folder (if subscription path belongs to "this"), 0 if missing;
  /// if new_request = true, send request to higher-level class if needed
  MonitorElementRootFolder * 
    getMonitorableFolder(const std::string & name, const DirFormat & dir,
						  bool new_request);
  /// remove objects from receiver's subscription list
  /// "name" format: see DirFormat definition
  /// return success flag
  bool removeSubscription(const std::string & name);
  /// send list of monitorables; return success flag
  bool sendMonitorable(void);
  /// send monitoring;  return # of monitoring objects sent (<0 for errors)
  int sendMonitoring(void);
  /// put updated monitoring objects into (private member) mess;
  /// return # of objects to be sent out
  int produceMonitoring(MonitorElementRootFolder * folder);
  
  /// check if requests from downstream class have been fullfilled
  void checkPendingRequests(void);

  /// copy monitoring elements (described in subscription request: <subsc>)
  /// from directory <dir_orig>; create new request <missing_items> with objects
  /// that were not found in <dir_orig>;
  /// return true if at least one ME was copied
  bool copy(const MonitorElementRootFolder * dir_orig, const DirFormat
	    & subsc, std::string & missing_items);

  /// come here to save subscription request for upstream class
  void saveRequest2add(const std::string & name);

  /// locally created message for sending stuff to nodes
  DQMMessage * mess_tmp;
  /// true for sources, false for everybody else
  bool pureProducer_; 
  /// true if we should send monitorable to receiver
  bool shouldSendMonitorable(void) const;

  /// true if ME in folder has never been sent to subscriber
  bool isNeverSent(MonitorElementRootFolder * folder, std::string ME_name);
  /// true if QReport for ME in folder has never been sent to subscriber
  bool isNeverSent(MonitorElementRootFolder * folder, std::string ME_name,
		   std::string qtname);
  /// set "neverSent" flag for ME in folder of subscriber
  void setNeverSent(MonitorElementRootFolder * folder, std::string ME_name, 
		    bool flag);
  /// set "neverSent" flag for QReport of ME in folder of subscriber
  void setNeverSent(MonitorElementRootFolder * folder, std::string ME_name, 
		    std::string qtname, bool flag);

  /// send TObjString from MERootQReport
  void sendTObjStringQ(MERootQReport * me, DQMRootBuffer *buffer);
  /// send TObjString from MonitorElement
  void sendTObjString(MonitorElement * me, DQMRootBuffer *buffer);

  /// send QReports associated with <me>; return total # of reports sent
  unsigned sendQReports(MonitorElement * me, MonitorElementRootFolder * folder,
			DQMRootBuffer *buffer);

 protected:
  /// time delay in between shipments (in microsecs)
  unsigned del_;
  /// message read-in by inheriting class
  DQMMessage * send_mess; 
  /// string read-in by inheriting class (only if mess->What() == kMESS_STRING)
  std::string * buffer_;
  /// this is really the socket of the receiver; 
  /// use "send" to emphasize that it is a protected member of the SenderBase class
  int send_socket;
  /// monitoring updates
  unsigned updates;

  struct ReceiverData_
  {
    /// subscriber directory structure with monitoring objects
    dqm::me_util::rootDir * Dir;
    /// name of receiver
    std::string name;
    /// # of monitoring packages sent
    unsigned count;
    /// # of failed attempts to send monitoring (and consecutive ones)
    unsigned n_failed, n_failed_consec;
    /// # of succesfull attempts to send monitoring (and timeouts)
    unsigned n_sent, n_timeout;
    /// true if we need to read subscription info
    bool need2readSubscription;
    /// true if node has not been sent full monitorable
    bool newNode;
    /// subscription requests to be forwarded to upstream class
    std::list<std::string> request2add;
    /// unsubscription requests to be forwarded to upstream class
    std::list<std::string> request2remove;
    /// pending subscription requests
    std::list<std::string> pend_request;
    /// default constructor
    ReceiverData_() {Dir = 0; count = 0; 
      name = dqm::monitor_data::DummyNodeName;
      need2readSubscription = false; newNode = true;
      n_failed = n_failed_consec = 0;
      n_sent = n_timeout = 0;
      request2add.clear(); request2remove.clear(); pend_request.clear();
    }
  };
  typedef struct ReceiverData_ ReceiverData;
    
  /// pointer to receiver's data (to be set by inheriting class)
  ReceiverData * receiver_;

  /// main "sending" loop: come here to send monitorable or monitoring
  /// return # of monitoring objects sent
  int send();
  /// come here to send monitorable; return success flag
  bool shipMonitorable();
  // come here to send monitoring; return # of objects sent
  int shipMonitoring();

  /// receive subscription from receiver
  void getSubscription(void); 
  /// make directory structure (for receiver; to be called by inheriting class)
  dqm::me_util::rootDir *  makeDirStructure(const std::string & name);

  /// cleanup methods to be called by inheriting class when node is disconnected
  /// come here for cleanup when a receiver goes down; do not release memory
  void cleanupReceiver(void); 
  /// come here for cleanup when sender goes down; do not release memory
  void cleanupSender(const std::string & sender_name);
  /// true if receiver is done sending subscription requests
  bool isReceiverDone(void) const;
 
  /** come here after sending monitoring to all receivers;
     (a) call resetUpdate for modified contents:
     
     if resetMEs=true, reset MEs that were updated (and have resetMe = true);
     [flag resetMe is typically set by sources (false by default)];
     [Clients in standalone mode should also have resetMEs = true] 
     
     (b) if callResetDiff = true, call resetMonitoringDiff
     (typical behaviour: Sources & Collector have callResetDiff = true, whereas
     clients have callResetDiff = false, so GUI/WebInterface can access the 
     modifications in monitorable & monitoring) */
  void doneSendingMonitoring(bool resetMEs, bool callResetDiff);
  /** come here after sending monitorable to all receivers;
     if callResetDiff = true, call resetMonitorableDiff
     (typical behaviour: Sources & Collector have callResetDiff = true, whereas
     clients have callResetDiff = false, so GUI/WebInterface can access the 
     modifications in monitorable & monitoring) */
  void doneSendingMonitorable(bool callResetDiff);
  /// doneSendingMonitorable & doneSendingMonitoring combined
  void doneSending(bool resetMEs, bool callResetDiff);
  /// come here at beginning of monitoring cycle
  void startSending(void);
  /// come here when attempt to send monitorable/monitoring fails
  void sendingFailed(void);
  /// check if objects appearing in unsubscription request <dir> are needed
  /// by any other subscribers (if any); if not, will issue unsubscription request
  /// to upstream class (if any)
  void checkIfNeeded(DirFormat & dir);
  /// prepare subscription requests for higher-level class (if applicable)
  void prepareRecvSubscription(void);
  /// send monitorable string, return success flag
  bool sendMonString(const std::string & sendThis);

};

#endif // _SENDER_BASE_H_
