#ifndef _RECEIVER_BASE_H_
#define _RECEIVER_BASE_H_

#include "DQMServices/Core/interface/MonitorElementRootT.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/src/NodeBase.h"
#include "DQMServices/Core/src/MonitorData.h"

#include <TString.h>
#include <TTimeStamp.h>

#include <string>
#include <vector>
#include <deque>

class DQMMessage;
class TObjString;
class TimeMonitor;

/** base class for clients that receive (any kind of) monitoring information; 
   the class organizes the objects it owns through the NodeBase::bei 
   (DaqMonitorBEInterface) interface; 
   (pointer sender_ is to be set by inheriting class)

   directory for saving monitoring:
   monitorName + <source name> + pathname of monitoring objects
*/
class ReceiverBase : public NodeBase
{
 public:
  /// node name
  ReceiverBase(std::string name, bool keepStaleSources = false);
  virtual ~ReceiverBase(void);
  
 private:
  /// prepare subscription request; name format: see DirFormat definition
  /// if flag=true: add subscription, else: cancel subscription;
  /// non-existing MEs are discarded from <name>
  void modifySubscription(std::string name, bool add);
  /// process string messages
  void processString(void);
  /// process object messages
  void processObject(void);
  /// get update time chart histogram
  void getTimeChartHist(void);
  /// get message size histogram for socket
  void getMessageSizeHist(float msize);
  /// get update frequency histogram for socket
  void getUpdFreqHist();

  /// extract object (TH1F, TH2F, ...) from message; return success flag
  bool extractObject(std::string pathname);
  /// add objects to monitorable; "name" format: see DirFormat definition
  /// return success flag
  bool addMonitorable(const std::string & name);
  /// remove objects from monitorable; "name" format: see DirFormat definition
  /// return success flag
  bool removeMonitorable(const std::string & name);
  /// get object description sent by Sender; format: see checkObjDesc
  void getObjectDesc(const std::string & name);
  /// loop over bei->request2add, request2remove: look for requests to sender_;
  /// when match found, update private members addMe, removeMe
  void produceSubscription(void);
  /// remove directory <subdir> from monitorable
  void lessMonitorable(const std::string subdir);
  /// add directory <subdir> to monitorable
  void moreMonitorable(const std::string subdir);
  /// return <sender's name>_is_dead
  std::string getDeadName(void) const;
  /// return <sender's name>_is_done
  std::string getDoneName(void) const;

  /// add tags to (addFlag=true) or remove tags from (addFlag = false) monitorable; 
  /// "name" format: see DirFormat definition
  /// (with exception that <obj> is replaced by <obj>/tag1/tag2, etc.
  /// return success flag
  bool modifyTags(const std::string & name, bool addFlag);

  /// set ME's "canDelete" property in directory <folder>;  
  /// to be used to set property to false when ME is extracted in this class
  void setCanDelete(MonitorElementRootFolder * folder, 
		    std::string ME_name, bool flag) const;
  /// call setCanDelete in current directory
  ///  void setCanDelete(std::string ME_name, bool flag) const;
  /// call setCanDelete for all ME in directory <folder>
  void setCanDelete(MonitorElementRootFolder * folder, bool flag) const;
  ///
  /// true if Monitoring Element <me> in directory <folder> has isDesired = true;
  /// if warning = true and <me> does not exist, show warning
  bool isDesired(MonitorElementRootFolder * folder, std::string me,
		 bool warning = false) const;
  /// set <name> ME's "isDesired" property in <folder>: to be used 
  /// to set property to true/false when ME is (un)subscribed in ReceiverBase class
  void setIsDesired(MonitorElementRootFolder * folder, 
		    std::string ME_name, bool flag) const;
  /// call setIsDesired for all ME in <dir>
  void setIsDesired(MonitorElementRootFolder * folder, bool flag) const;

  /// remove directory, update monitorable
  void removeDir(std::string subdir);

  /// update time chart
  TH1F *timechart;

  // TimeMonitor used to perform time mesaurements
  TimeMonitor *timeMonitor;
    
  std::deque<float> vdelay;
  float meanupdel;
  float updelds;
  float tx;

 protected:

  /// structure holding TSocket statistics/data
  struct SenderData_ {
    /// connection name
    std::string name;
    /// # of objects to be read
    unsigned objn;
    /// # of monitoring packages received
    unsigned count;
    /// current pathname
    std::string path;
    /// histogram with message sizes
    TH1F * size;
    /// histogram with update frequency
    TH1F * freq;
    /// timer
    TTimeStamp timer;
    /// flags for
    // whether there is new subscription info
    // bool newSubscription; 
    /// whether need to read in modified monitorable
    bool need2readMonitorable;
    /// whether need to read in monitoring info
    bool need2readMonitoring;  
    /// new subscriptions (to be sent to "Sender");
    /// contain names of format described in DirFormat
    std::vector<std::string> addMe;
    /// cancell subscriptions (to be sent to "Sender");
    /// contain names of format described in DirFormat
    std::vector<std::string> removeMe;
    /// # of full monitoring cycles waiting to be sent out
    unsigned cycles_count;
    /// default constructor
    SenderData_() {name = dqm::monitor_data::DummyNodeName; 
      objn=count = 0; path ="unknown_path"; 
      size = freq = 0; timer = TTimeStamp(); 
      need2readMonitorable= need2readMonitoring = false;
      addMe.clear(); removeMe.clear(); cycles_count = 0;
    }
  };
  typedef struct SenderData_ SenderData;

  /// # of monitoring updates
  unsigned updates;
  /// # of monitoring elements received in last message
  unsigned N_me_recv;
  /// string read-in by inheriting class (only if mess->What() == kMESS_STRING)
  std::string * buffer_;

  DQMMessage * recv_mess;
  /// this is really the socket of the sender; 
  ///use "recv" to emphasize that it is a protected member of the ReceiverBase class
  int recv_socket;

  /// pointer to sender's SenderData (to be set by inheriting class)
  SenderData * sender_;

  /// if true, will keep monitoring structure in memory
  /// when sender goes down/finishes processing (default: false)
  bool keepStaleSources_;

  /// main "receiving" loop: come here once message has been received 
  /// (in higher-level class)
  /// return kMESS_DONE_MONIT_CYCLE if done w/ monitoring cycle, 0 otherwise
  int receive(void);

  /// send subscription to sender
  void sendSubscription(void);

  /// remove monitoring elements and release memory associated with sender_; 
  /// to be called by inheriting class when client is disconnected
  void cleanupSender(void);
  /// remove monitoring elements associated with (obsolete) sender_(with same name);
  /// to be called by inheriting class when sender is (re)connected
  void cleanupObsoleteSender(void);
  /// to be called by cleanupSender and cleanupObsoleteSender
  void cleanup(std::string name, bool doIt = true);

  /// save output file
  void saveFile(const std::string & filename);

  /// true if there are new (un)subscription requests
  bool canSubscribe(void) const;

  /// true if client should send subscription
  bool shouldSendSubscription(void)
  {
    if(!sender_)return false;
    // should send subscription request if (a) there is new subscription info
    // and (b) sender is done 
    bool newSubsInfo = (!sender_->addMe.empty() || 
		     !sender_->removeMe.empty());
    return ( newSubsInfo && isSenderDone());
  }

  /// true if sender is done sending (a) monitorable & (b) monitoring
  bool isSenderDone(void) const;

  /// main subscription loop
  void doSubscription(void);

  /// true if there is updated monitorable
  bool newMonitorable(void) const;

  /// create string at top directory indicating sender is dead
  void senderIsDead(void);
  /// create string at top directory indicating sender is done
  void senderIsDone(void);
  /// reverse action of senderIsDead
  void senderIsNotDead(void);
  /// reverse action of senderIsDone
  void senderIsNotDone(void);

};

#endif // _RECEIVER_BASE_H_
