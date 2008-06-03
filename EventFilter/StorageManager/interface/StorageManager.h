#ifndef _StorageManager_h
#define _StorageManager_h

/*
   Author: Harry Cheung, FNAL

   Description:
     Storage Manager XDAQ application. It can receive and collect
     I2O frames to remake event data. 

     See CMS EventFilter wiki page for further notes.

   $Id: StorageManager.h,v 1.32 2008/05/13 18:06:46 loizides Exp $
*/

#include <string>
#include <list>
#include <map>

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageService/interface/MessageServicePresence.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/Css.h"
#include "EventFilter/Utilities/interface/RunBase.h"
#include "EventFilter/Utilities/interface/StateMachine.h"

#include "EventFilter/StorageManager/interface/JobController.h"
#include "EventFilter/StorageManager/interface/SMPerformanceMeter.h"
#include "EventFilter/StorageManager/interface/SMFUSenderList.h"

#include "FWCore/PluginManager/interface/PluginManager.h"

#include "toolbox/mem/Reference.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"

#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Integer.h"
#include "xdata/Double.h"
#include "xdata/Boolean.h"
#include "xdata/Vector.h"

#include "xgi/Input.h"
#include "xgi/Output.h"
#include "xgi/exception/Exception.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"


namespace stor {

  class StorageManager: public xdaq::Application, 
                        public xdata::ActionListener,
                        public evf::RunBase
  {
   public:
    StorageManager(xdaq::ApplicationStub* s) throw (xdaq::exception::Exception);
  
    ~StorageManager();

    // *** Updates the exported parameters
    xoap::MessageReference ParameterGet(xoap::MessageReference message)
    throw (xoap::exception::Exception);

    // *** Anything to do with the flash list
    void setupFlashList();
    void actionPerformed(xdata::Event& e);

    // *** Callbacks to be executed during transitional states
    bool configuring(toolbox::task::WorkLoop* wl);
    bool enabling(toolbox::task::WorkLoop* wl);
    bool stopping(toolbox::task::WorkLoop* wl);
    bool halting(toolbox::task::WorkLoop* wl);

    // *** FSM soap command callback
    xoap::MessageReference fsmCallback(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
    // @@EM added monitoring workloop
    void startMonitoringWorkLoop() throw (evf::Exception);
    bool monitoring(toolbox::task::WorkLoop* wl);
    
////////////////////////////////////////////////////////////////////////////////
   private:  
    void receiveRegistryMessage(toolbox::mem::Reference *ref);
    void receiveDataMessage(toolbox::mem::Reference *ref);
    void receiveErrorDataMessage(toolbox::mem::Reference *ref);
    void receiveOtherMessage(toolbox::mem::Reference *ref);
    void receiveDQMMessage(toolbox::mem::Reference *ref);

    void sendDiscardMessage(unsigned int, 
			    unsigned int, 
			    unsigned int, 
			    std::string);

    void stopAction();
    void haltAction();

    void checkDirectoryOK(const std::string dir) const;

    void defaultWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void css(xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception)
      {css_.css(in,out);}
    void fusenderWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void streamerOutputWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void eventdataWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void headerdataWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void consumerWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void consumerListWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void eventServerWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void DQMeventdataWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void DQMconsumerWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);


    void parseFileEntry(const std::string &in, std::string &out, unsigned int &nev, unsigned long long &sz) const;

    std::string findStreamName(const std::string &in) const;
	
    // *** state machine related
    evf::StateMachine fsm_;
    std::string       reasonForFailedState_;

    edm::AssertHandler *ah_;
    edm::service::MessageServicePresence theMessageServicePresence;
    xdata::String offConfig_;
  
    boost::shared_ptr<stor::JobController> jc_;
    boost::mutex                           halt_lock_;

    xdata::Boolean pushmode2proxy_;
    xdata::Integer nLogicalDisk_;
    xdata::String  fileName_;
    xdata::String  filePath_;
    xdata::Integer maxFileSize_;
    xdata::String  setupLabel_;
    xdata::Double  highWaterMark_;
    xdata::Double  lumiSectionTimeOut_;
    xdata::String  fileCatalog_;
    xdata::Boolean exactFileSizeTest_;

    bool pushMode_;
    std::string smConfigString_;
    std::string smFileCatalog_;

    xdata::Boolean collateDQM_;
    xdata::Boolean archiveDQM_;
    xdata::String  filePrefixDQM_;
    xdata::Integer purgeTimeDQM_;
    xdata::Integer readyTimeDQM_;
    xdata::Boolean useCompressionDQM_;
    xdata::Integer compressionLevelDQM_;

    evf::Css css_;
    xdata::UnsignedInteger32 receivedFrames_;
    int pool_is_set_;
    toolbox::mem::Pool *pool_;

    // added for Event Server
    std::vector<unsigned char> mybuffer_; //temporary buffer instead of using stack
    xdata::Double maxESEventRate_;  // hertz
    xdata::Double maxESDataRate_;  // MB/sec
    xdata::Integer activeConsumerTimeout_;  // seconds
    xdata::Integer idleConsumerTimeout_;  // seconds
    xdata::Integer consumerQueueSize_;
    xdata::Boolean fairShareES_;
    xdata::Double DQMmaxESEventRate_;  // hertz
    xdata::Integer DQMactiveConsumerTimeout_;  // seconds
    xdata::Integer DQMidleConsumerTimeout_;  // seconds
    xdata::Integer DQMconsumerQueueSize_;
    boost::mutex consumerInitMsgLock_;

    SMFUSenderList smfusenders_;
    xdata::UnsignedInteger32 connectedFUs_;

    xdata::UnsignedInteger32 storedEvents_;
    xdata::UnsignedInteger32 receivedEvents_;
    xdata::UnsignedInteger32 receivedErrorEvents_;
    xdata::UnsignedInteger32 dqmRecords_;
    xdata::UnsignedInteger32 closedFiles_;
    xdata::UnsignedInteger32 openFiles_;
    xdata::Vector<xdata::String> fileList_;
    xdata::Vector<xdata::UnsignedInteger32> eventsInFile_;
    xdata::Vector<xdata::UnsignedInteger32> storedEventsInStream_;
    xdata::Vector<xdata::UnsignedInteger32> receivedEventsFromOutMod_;
    typedef std::map<std::string,uint32> countMap;
    typedef std::map<uint32,std::string> idMap;
    typedef std::map<uint32,std::string>::iterator idMap_iter;
    countMap receivedEventsMap_;
    idMap modId2ModOutMap_;
    countMap storedEventsMap_;
    xdata::Vector<xdata::UnsignedInteger32> fileSize_;
    xdata::Vector<xdata::String> namesOfStream_;
    xdata::Vector<xdata::String> namesOfOutMod_;

    // *** for performance measurements
    void addMeasurement(unsigned long size);
    stor::SMPerformanceMeter *pmeter_;

    // *** measurements for last set of samples
    xdata::UnsignedInteger32 samples_; // number of samples/frames per measurement
    xdata::Double instantBandwidth_; // bandwidth in MB/s
    xdata::Double instantRate_;      // number of frames/s
    xdata::Double instantLatency_;   // micro-seconds/frame
    xdata::Double maxBandwidth_;     // maximum bandwidth in MB/s
    xdata::Double minBandwidth_;     // minimum bandwidth in MB/s

    // *** measurements for all samples
    xdata::Double duration_;         // time for run in seconds
    xdata::UnsignedInteger32 totalSamples_; //number of samples/frames per measurement
    xdata::Double meanBandwidth_;    // bandwidth in MB/s
    xdata::Double meanRate_;         // number of frames/s
    xdata::Double meanLatency_;      // micro-seconds/frame

    // *** additional flashlist contents (rest was already there)
    xdata::String            class_;
    xdata::UnsignedInteger32 instance_;
    xdata::String            url_;       

    xdata::Double            storedVolume_;
    xdata::UnsignedInteger32 memoryUsed_;
    xdata::String            progressMarker_;

    // @@EM workloop / action signature for monitoring
    toolbox::task::WorkLoop         *wlMonitoring_;      
    toolbox::task::ActionSignature  *asMonitoring_;

    // @@EM parameters monitored by workloop (not in flashlist just yet) 
    struct streammon{
      int		nclosedfiles_;
      int		nevents_;
      int		totSizeInkBytes_;
    };

    typedef std::map<std::string,streammon> smap;
    typedef std::map<std::string,streammon>::iterator ismap;
    smap	 streams_;

    unsigned int lastEventSeen_; // report last seen event id
    unsigned int lastErrorEventSeen_; // report last error event id seen
    boost::mutex fulist_lock_;  // quick (temporary) fix for registration problem

    enum
    {
      DEFAULT_PURGE_TIME = 120,
      DEFAULT_READY_TIME = 10
    };

  }; 
} 

#endif
