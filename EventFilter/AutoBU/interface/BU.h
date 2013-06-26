#ifndef AUTOBU_BU_H
#define AUTOBU_BU_H 1


#include "EventFilter/AutoBU/interface/BUEvent.h"

#include "EventFilter/Utilities/interface/StateMachine.h"
#include "EventFilter/Utilities/interface/WebGUI.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "EventFilter/Playback/interface/PlaybackRawDataProvider.h"

#include "xdaq/Application.h"

#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/net/URN.h"
#include "toolbox/fsm/exception/Exception.h"

#include "xdata/InfoSpace.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Double.h"
#include "xdata/Boolean.h"
#include "xdata/String.h"

#include "interface/evb/i2oEVBMsgs.h"
#include "interface/shared/i2oXFunctionCodes.h"

#include "interface/shared/frl_header.h"
#include "interface/shared/fed_header.h"
#include "interface/shared/fed_trailer.h"

#include "i2o/Method.h"
#include "i2o/utils/AddressMap.h"

#include "CLHEP/Random/RandGauss.h"


#include <vector>
#include <queue>
#include <cmath>
#include <semaphore.h>
#include <sys/time.h>


namespace evf {


  class BU : public xdaq::Application,
	     public xdata::ActionListener
  {
  public:
    //
    // xdaq instantiator macro
    //
    XDAQ_INSTANTIATOR();
  
    
    //
    // construction/destruction
    //
    BU(xdaq::ApplicationStub *s);
    virtual ~BU();
  
  
    //
    // public member functions
    //

    // work loop functions to be executed during transitional states (async)
    bool configuring(toolbox::task::WorkLoop* wl);
    bool enabling(toolbox::task::WorkLoop* wl);
    bool stopping(toolbox::task::WorkLoop* wl);
    bool halting(toolbox::task::WorkLoop* wl);
    
    // fsm soap command callback
    xoap::MessageReference fsmCallback(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
    
    // i2o callbacks
    void I2O_BU_ALLOCATE_Callback(toolbox::mem::Reference *bufRef) throw (i2o::exception::Exception);
    void I2O_BU_DISCARD_Callback(toolbox::mem::Reference *bufRef) throw (i2o::exception::Exception);
    
    // xdata::ActionListener callback
    void actionPerformed(xdata::Event& e);

    // Hyper DAQ web interface [see Utilities/WebGUI]
    void webPageRequest(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void customWebPage(xgi::Input*in,xgi::Output*out)
      throw (xgi::exception::Exception);
    
    // build events (random or playback)
    void startBuildingWorkLoop() throw (evf::Exception);
    bool building(toolbox::task::WorkLoop* wl);

    // send events to connected FU
    void startSendingWorkLoop() throw (evf::Exception);
    bool sending(toolbox::task::WorkLoop* wl);

    // calculate monitoring information in separate thread
    void startMonitoringWorkLoop() throw (evf::Exception);
    bool monitoring(toolbox::task::WorkLoop* wl);
    

  private:
    //
    // private member functions
    //
    void   lock()      { sem_wait(&lock_); }
    void   unlock()    { sem_post(&lock_); }
    void   waitBuild() { sem_wait(&buildSem_); }
    void   postBuild() { sem_post(&buildSem_); }
    void   waitSend()  { sem_wait(&sendSem_); }
    void   postSend()  { sem_post(&sendSem_); }
    void   waitRqst()  { sem_wait(&rqstSem_); }
    void   postRqst()  { sem_post(&rqstSem_); }
    
    void   exportParameters();
    void   reset();
    double deltaT(const struct timeval *start,const struct timeval *end);
    
    bool   generateEvent(evf::BUEvent* evt);
    toolbox::mem::Reference *createMsgChain(evf::BUEvent *evt,
					    unsigned int fuResourceId);
    
    
    void dumpFrame(unsigned char* data,unsigned int len);
  
  
  private:
    //
    // member data
    //

    // BU message logger
    Logger                          log_;

    // BU application descriptor
    xdaq::ApplicationDescriptor    *buAppDesc_;
    
    // FU application descriptor
    xdaq::ApplicationDescriptor    *fuAppDesc_;
    
    // BU application context
    xdaq::ApplicationContext       *buAppContext_;
    
    // BU state machine
    StateMachine                    fsm_;
    
    // BU web interface
    WebGUI                         *gui_;
    
    // resource management
    std::vector<evf::BUEvent*>      events_;
    std::queue<unsigned int>        rqstIds_;
    std::queue<unsigned int>        freeIds_;
    std::queue<unsigned int>        builtIds_;
    std::set<unsigned int>          sentIds_;
    unsigned int                    evtNumber_;
    std::vector<unsigned int>       validFedIds_;

    bool                            isBuilding_;
    bool                            isSending_;
    bool                            isHalting_;

    // workloop / action signature for building events
    toolbox::task::WorkLoop        *wlBuilding_;      
    toolbox::task::ActionSignature *asBuilding_;
    
    // workloop / action signature for sending events
    toolbox::task::WorkLoop        *wlSending_;      
    toolbox::task::ActionSignature *asSending_;
    
    // workloop / action signature for monitoring
    toolbox::task::WorkLoop        *wlMonitoring_;      
    toolbox::task::ActionSignature *asMonitoring_;
    
    
    std::string                     sourceId_;
        
    // monitored parameters
    xdata::String                   url_;
    xdata::String                   class_;
    xdata::UnsignedInteger32        instance_;
    xdata::String                   hostname_;
    xdata::UnsignedInteger32        runNumber_;
    xdata::Double                   memUsedInMB_;

    xdata::Double                   deltaT_;
    xdata::UnsignedInteger32        deltaN_;
    xdata::Double                   deltaSumOfSquares_;
    xdata::UnsignedInteger32        deltaSumOfSizes_;

    xdata::Double                   throughput_;
    xdata::Double                   average_;
    xdata::Double                   rate_;
    xdata::Double                   rms_;
    
    // monitored counters
    xdata::UnsignedInteger32        nbEventsInBU_;
    xdata::UnsignedInteger32        nbEventsRequested_;
    xdata::UnsignedInteger32        nbEventsBuilt_;
    xdata::UnsignedInteger32        nbEventsSent_;
    xdata::UnsignedInteger32        nbEventsDiscarded_;
    
    // standard parameters
    xdata::String                   mode_;
    xdata::Boolean                  replay_;
    xdata::Boolean                  crc_;
    xdata::Boolean                  overwriteEvtId_;
    xdata::Boolean                  overwriteLsId_;
    xdata::UnsignedInteger32        fakeLsUpdateSecs_;
    xdata::UnsignedInteger32        firstEvent_;
    xdata::UnsignedInteger32        queueSize_;
    xdata::UnsignedInteger32        eventBufferSize_;
    xdata::UnsignedInteger32        msgBufferSize_;
    xdata::UnsignedInteger32        fedSizeMax_;
    xdata::UnsignedInteger32        fedSizeMean_;
    xdata::UnsignedInteger32        fedSizeWidth_;
    xdata::Boolean                  useFixedFedSize_;
    xdata::UnsignedInteger32        monSleepSec_;

    unsigned int                    fakeLs_;
    timeval                         lastLsUpdate_;
    // gaussian aprameters for randpm fed size generation (log-normal)
    double                          gaussianMean_;
    double                          gaussianWidth_;
    
    // monitoring helpers
    struct timeval                  monStartTime_;
    unsigned int                    monLastN_;
    uint64_t                        monLastSumOfSquares_;
    unsigned int                    monLastSumOfSizes_;
    uint64_t                        sumOfSquares_;
    unsigned int                    sumOfSizes_;
    

    // memory pool for i20 communication
    toolbox::mem::Pool*             i2oPool_;

    // synchronization
    sem_t                           lock_;
    sem_t                           buildSem_;
    sem_t                           sendSem_;
    sem_t                           rqstSem_;

  
    //
    // static member data
    //
    static const int frlHeaderSize_ =sizeof(frlh_t);
    static const int fedHeaderSize_ =sizeof(fedh_t);
    static const int fedTrailerSize_=sizeof(fedt_t);
  
  }; // class BU


} // namespace evf


#endif
