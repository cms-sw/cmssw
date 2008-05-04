#ifndef _SMProxyServer_h
#define _SMProxyServer_h

/*
   Description:
     Storage Manager Proxy Server XDAQ application. 
     This receives event data and DQM data from the
     Storage Manager of each subfarm. The event data is meant
     to be limited to up to a few hertz to serve to event consumers.
     The DQM data from all FUs of one update is collated and then
     written to disk and also available to send to DQM consumers.

     See CMS EventFilter wiki page for further notes.

   $Id: SMProxyServer.h,v 1.9 2008/04/16 18:25:18 biery Exp $
*/

#include <string>
#include <list>

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageService/interface/MessageServicePresence.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/Utilities/interface/Css.h"
#include "EventFilter/Utilities/interface/RunBase.h"
#include "EventFilter/Utilities/interface/StateMachine.h"

#include "EventFilter/SMProxyServer/interface/DataProcessManager.h"
#include "EventFilter/StorageManager/interface/SMPerformanceMeter.h"
#include "EventFilter/StorageManager/interface/SMFUSenderList.h"

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

  class SMProxyServer: public xdaq::Application, 
                        public xdata::ActionListener,
                        public evf::RunBase
  {
   public:
    SMProxyServer(xdaq::ApplicationStub* s) throw (xdaq::exception::Exception);
  
    ~SMProxyServer();

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

    
////////////////////////////////////////////////////////////////////////////////
   private:  
    void checkDirectoryOK(std::string dir);

    void defaultWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void css(xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception)
      {css_.css(in,out);}
    void smsenderWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void DQMOutputWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void eventdataWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void headerdataWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void consumerWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void eventServerWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void DQMeventdataWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void DQMconsumerWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);

    void receiveEventWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);
    void receiveDQMEventWebPage
      (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);

    evf::StateMachine fsm_;
    std::string reasonForFailedState_;

    edm::AssertHandler *ah_;
    edm::service::MessageServicePresence theMessageServicePresence;
  
    boost::shared_ptr<stor::DataProcessManager> dpm_;
    boost::mutex                           halt_lock_;

    //xdata::Integer nLogicalDisk_;
    //xdata::String  fileCatalog_;

    //xdata::String  closeFileScript_;
    //xdata::String  notifyTier0Script_;
    //xdata::String  insertFileScript_;

    //std::string smFileCatalog_;

    xdata::Boolean collateDQM_;
    xdata::Boolean archiveDQM_;
    xdata::String  filePrefixDQM_;
    xdata::Integer purgeTimeDQM_;
    xdata::Integer readyTimeDQM_;
    xdata::Boolean useCompressionDQM_;
    xdata::Integer compressionLevelDQM_;

    evf::Css css_;
    xdata::UnsignedInteger32 receivedEvents_;
    xdata::UnsignedInteger32 receivedDQMEvents_;

    // for Event Server
    std::vector<unsigned char> mybuffer_;
    xdata::Vector<xdata::String> smRegList_; // StorageManagers to subscribe to
    xdata::String consumerName_;
    xdata::String DQMconsumerName_;

    xdata::Double maxESEventRate_;  // hertz
    xdata::Double maxESDataRate_;  // hertz
    xdata::Double maxEventRequestRate_;  // hertz
    xdata::Integer activeConsumerTimeout_;  // seconds
    xdata::Integer idleConsumerTimeout_;  // seconds
    xdata::Integer consumerQueueSize_;
    xdata::Boolean fairShareES_;
    xdata::Double DQMmaxESEventRate_;  // hertz
    xdata::Double maxDQMEventRequestRate_;  // hertz
    xdata::Integer DQMactiveConsumerTimeout_;  // seconds
    xdata::Integer DQMidleConsumerTimeout_;  // seconds
    xdata::Integer DQMconsumerQueueSize_;

    std::map< std::string, bool > smsenders_;
    xdata::UnsignedInteger32 connectedSMs_;

    xdata::UnsignedInteger32 storedDQMEvents_;
    xdata::UnsignedInteger32 sentEvents_;
    xdata::UnsignedInteger32 sentDQMEvents_;
    //xdata::UnsignedInteger32 closedFiles_;
    //xdata::Vector<xdata::String> fileList_;
    //xdata::Vector<xdata::UnsignedInteger32> eventsInFile_;
    //xdata::Vector<xdata::UnsignedInteger32> fileSize_;

    // *** for performance measurements
    void addMeasurement(unsigned long size);
    void addOutMeasurement(unsigned long size);
    stor::SMPerformanceMeter *outpmeter_;

    // *** measurements for last set of samples
    xdata::UnsignedInteger32 samples_; // number of samples/frames per measurement
    xdata::Double instantBandwidth_; // bandwidth in MB/s
    xdata::Double instantRate_;      // number of frames/s
    xdata::Double instantLatency_;   // micro-seconds/frame
    xdata::Double maxBandwidth_;     // maximum bandwidth in MB/s
    xdata::Double minBandwidth_;     // minimum bandwidth in MB/s
    xdata::Double outinstantBandwidth_; // bandwidth in MB/s
    xdata::Double outinstantRate_;      // number of frames/s
    xdata::Double outinstantLatency_;   // micro-seconds/frame
    xdata::Double outmaxBandwidth_;     // maximum bandwidth in MB/s
    xdata::Double outminBandwidth_;     // minimum bandwidth in MB/s

    // *** measurements for all samples
    xdata::Double duration_;         // time for run in seconds
    xdata::UnsignedInteger32 totalSamples_; //number of samples/frames per measurement
    xdata::Double meanBandwidth_;    // bandwidth in MB/s
    xdata::Double meanRate_;         // number of frames/s
    xdata::Double meanLatency_;      // micro-seconds/frame
    xdata::Double outduration_;         // time for run in seconds
    xdata::UnsignedInteger32 outtotalSamples_; //number of samples/frames per measurement
    xdata::Double outmeanBandwidth_;    // bandwidth in MB/s
    xdata::Double outmeanRate_;         // number of frames/s
    xdata::Double outmeanLatency_;      // micro-seconds/frame

    // *** additional flashlist contents (rest was already there)
    xdata::String            class_;
    xdata::UnsignedInteger32 instance_;
    xdata::String            url_;       

    xdata::Double            storedVolume_;
    xdata::UnsignedInteger32 memoryUsed_;
    xdata::String            progressMarker_;
    enum
    {
      DEFAULT_PURGE_TIME = 20,
      DEFAULT_READY_TIME = 10
    };
  }; 
} 


#endif
