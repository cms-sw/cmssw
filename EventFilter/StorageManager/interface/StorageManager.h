#ifndef _StorageManager_h
#define _StorageManager_h

/*
   Author: Harry Cheung, FNAL

   Description:
     Storage Manager XDAQ application. It can receive and collect
     I2O frames to remake event data. 

     See CMS EventFilter wiki page for further notes.
*/

// $Id$

#include <string>
#include <list>

#include "FWCore/Utilities/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageService/interface/MessageServicePresence.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/Utilities/interface/Css.h"
#include "EventFilter/Utilities/interface/RunBase.h"
#include "EventFilter/StorageManager/interface/JobController.h"
#include "EventFilter/StorageManager/interface/SMPerformanceMeter.h"
#include "EventFilter/StorageManager/interface/SMFUSenderList.h"
#include "EventFilter/StorageManager/interface/SMStateMachine.h"

#include "PluginManager/PluginManager.h"

#include "toolbox/mem/Reference.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"

#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Integer.h"
#include "xdata/Double.h"
#include "xdata/Boolean.h"
#include "xdata/Vector.h"

#include "xgi/include/xgi/Input.h"
#include "xgi/include/xgi/Output.h"
#include "xgi/include/xgi/exception/Exception.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"


namespace stor {

  class StorageManager: public xdaq::Application, xdata::ActionListener,
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
  
   private:
    void configureAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception);
    void enableAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception);
    virtual void haltAction(toolbox::Event::Reference e) 
      throw (toolbox::fsm::exception::Exception);
    virtual void nullAction(toolbox::Event::Reference e) 
      throw (toolbox::fsm::exception::Exception);
    xoap::MessageReference fireEvent(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
  
    void receiveRegistryMessage(toolbox::mem::Reference *ref);
    void receiveDataMessage(toolbox::mem::Reference *ref);
    void receiveOtherMessage(toolbox::mem::Reference *ref);

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

    void parseFileEntry(std::string in, std::string &out, unsigned int &nev, unsigned int &sz);
	
    stor::SMStateMachine *fsm_;
    edm::AssertHandler *ah_;
    edm::service::MessageServicePresence theMessageServicePresence;
    xdata::String offConfig_;
 
    friend class stor::SMStateMachine;
  
    boost::shared_ptr<stor::JobController> jc_;
    boost::mutex                           halt_lock_;

    // added for streamer file writing instead of OutServ
    xdata::Boolean streamer_only_;
    xdata::Integer nLogicalDisk_;
    xdata::String  fileCatalog_;

    bool writeStreamerOnly_;
    std::string smConfigString_;
    std::string smFileCatalog_;

    evf::Css css_;
    xdata::UnsignedInteger32 receivedFrames_;
    int pool_is_set_;
    toolbox::mem::Pool *pool_;

    // added temporarily for Event Server
    char serialized_prods_[1000*1000*2];
    int  ser_prods_size_;
    xdata::Integer oneinN_; //place one in eveny oneinN_ into buffer
    char mybuffer_[7000000]; //temporary buffer instead of using stack
    xdata::Double maxESEventRate_;  // hertz
    xdata::Integer activeConsumerTimeout_;  // seconds
    xdata::Integer idleConsumerTimeout_;  // seconds
    xdata::Integer consumerQueueSize_;

    SMFUSenderList smfusenders_;
    xdata::UnsignedInteger32 connectedFUs_;

    xdata::UnsignedInteger32 storedEvents_;
    xdata::UnsignedInteger32 closedFiles_;
    xdata::Vector<xdata::String> fileList_;
    xdata::Vector<xdata::UnsignedInteger32> eventsInFile_;
    xdata::Vector<xdata::UnsignedInteger32> fileSize_;

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
  }; 
} 


#endif
