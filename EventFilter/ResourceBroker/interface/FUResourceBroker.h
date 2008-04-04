#ifndef FURESOURCEBROKER_H
#define FURESOURCEBROKER_H 1


#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/ResourceBroker/interface/FUResourceTable.h"

#include "EventFilter/Utilities/interface/StateMachine.h"
#include "EventFilter/Utilities/interface/WebGUI.h"

#include "xdaq/Application.h"
#include "xdaq/NamespaceURI.h"

#include "xdata/InfoSpace.h"
#include "xdata/String.h"
#include "xdata/Boolean.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Double.h"

#include "toolbox/mem/Reference.h"
#include "toolbox/fsm/exception/Exception.h"

#include "interface/shared/frl_header.h"
#include "interface/shared/fed_header.h"
#include "interface/shared/fed_trailer.h"

#include <vector>
#include <string>
#include <semaphore.h>
#include <sys/time.h>


namespace evf {

  class BUProxy;
  class SMProxy;
  
  class FUResourceBroker : public xdaq::Application,
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
    FUResourceBroker(xdaq::ApplicationStub *s);
    virtual ~FUResourceBroker();
    
    
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
    
    // i20 callbacks
    void I2O_FU_TAKE_Callback(toolbox::mem::Reference *bufRef);
    void I2O_FU_DATA_DISCARD_Callback(toolbox::mem::Reference *bufRef);
    void I2O_FU_DQM_DISCARD_Callback(toolbox::mem::Reference *bufRef);

    //  connection to BuilderUnit bu_ and StorageManager sm_
    void connectToBUandSM() throw (evf::Exception);
    
    // Hyper DAQ web page(s) [see Utilities/WebGUI]
    void webPageRequest(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void customWebPage(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    
    // xdata::ActionListener callback(s)
    void actionPerformed(xdata::Event& e);
    
    // calculate monitoring information in separate thread
    void startMonitoringWorkLoop() throw (evf::Exception);
    bool monitoring(toolbox::task::WorkLoop* wl);
    
    // watch the state of the shm buffer in a separate thread
    void startWatchingWorkLoop() throw (evf::Exception);
    bool watching(toolbox::task::WorkLoop* wl);
    
    
  private:
    //
    // private member functions
    //
    void   exportParameters();
    void   reset();
    double deltaT(const struct timeval *start,const struct timeval *end);
    
    
  private:    
    //
    // member data
    //
    
    // finite state machine
    evf::StateMachine        fsm_;
    
    // Hyper DAQ web GUI
    WebGUI*                  gui_;
    
    // application logger
    Logger                   log_;
    
    // BuilderUnit (BU) to receive raw event data from
    BUProxy                 *bu_;
    
    // StorageManager (SM) to send selected events to
    SMProxy                 *sm_;
    
    // memory pool for bu <-> fu comunication messages
    toolbox::mem::Pool*      i2oPool_;
    
    // managed resources
    FUResourceTable*         resourceTable_;
    
    // workloop / action signature for monitoring
    toolbox::task::WorkLoop *wlMonitoring_;      
    toolbox::task::ActionSignature *asMonitoring_;
    
    // workloop / action signature for watching
    toolbox::task::WorkLoop *wlWatching_;      
    toolbox::task::ActionSignature *asWatching_;
    
    // application identifier
    std::string              sourceId_;
    
    // monitored parameters 
    xdata::String            url_;
    xdata::String            class_;
    xdata::UnsignedInteger32 instance_;
    xdata::UnsignedInteger32 runNumber_;

    
    xdata::Double            deltaT_;
    xdata::UnsignedInteger32 deltaNbInput_;
    xdata::UnsignedInteger32 deltaNbOutput_;
    xdata::Double            deltaInputSumOfSquares_;
    xdata::Double            deltaOutputSumOfSquares_;
    xdata::UnsignedInteger32 deltaInputSumOfSizes_;
    xdata::UnsignedInteger32 deltaOutputSumOfSizes_;
    
    xdata::Double            ratio_;
    xdata::Double            inputThroughput_;
    xdata::Double            inputRate_;
    xdata::Double            inputAverage_;
    xdata::Double            inputRms_;
    xdata::Double            outputThroughput_;
    xdata::Double            outputRate_;
    xdata::Double            outputAverage_;
    xdata::Double            outputRms_;
    
    // monitored counters
    xdata::UnsignedInteger32 nbAllocatedEvents_;
    xdata::UnsignedInteger32 nbPendingRequests_;
    xdata::UnsignedInteger32 nbReceivedEvents_;
    xdata::UnsignedInteger32 nbProcessedEvents_;
    xdata::UnsignedInteger32 nbAcceptedEvents_;
    xdata::UnsignedInteger32 nbSentEvents_;
    xdata::UnsignedInteger32 nbDiscardedEvents_;
    
    xdata::UnsignedInteger32 nbLostEvents_;
    xdata::UnsignedInteger32 nbDataErrors_;
    xdata::UnsignedInteger32 nbCrcErrors_;
    
    // standard parameters
    xdata::Boolean           segmentationMode_;
    xdata::UnsignedInteger32 nbClients_;
    xdata::String            clientPrcIds_;
    xdata::UnsignedInteger32 nbRawCells_;
    xdata::UnsignedInteger32 nbRecoCells_;
    xdata::UnsignedInteger32 nbDqmCells_;
    xdata::UnsignedInteger32 rawCellSize_;
    xdata::UnsignedInteger32 recoCellSize_;
    xdata::UnsignedInteger32 dqmCellSize_;
    
    xdata::Boolean           doDropEvents_;
    xdata::Boolean           doFedIdCheck_;
    xdata::UnsignedInteger32 doCrcCheck_;
    xdata::UnsignedInteger32 doDumpEvents_;

    xdata::String            buClassName_;
    xdata::UnsignedInteger32 buInstance_;
    xdata::String            smClassName_;
    xdata::UnsignedInteger32 smInstance_;
    
    xdata::UnsignedInteger32 monSleepSec_;
    xdata::UnsignedInteger32 watchSleepSec_;
    xdata::UnsignedInteger32 timeOutSec_;
    xdata::Boolean           processKillerEnabled_;
    xdata::Boolean           useEvmBoard_;

    xdata::String            reasonForFailed_;
    

    // debug parameters
    xdata::UnsignedInteger32 nbAllocateSent_;
    xdata::UnsignedInteger32 nbTakeReceived_;
    xdata::UnsignedInteger32 nbDataDiscardReceived_;
    xdata::UnsignedInteger32 nbDqmDiscardReceived_;
    
    // helper variables for monitoring
    struct timeval           monStartTime_;
    UInt_t                   nbInputLast_;
    uint64_t                 nbInputLastSumOfSquares_;
    UInt_t                   nbInputLastSumOfSizes_;
    UInt_t                   nbOutputLast_;
    uint64_t                 nbOutputLastSumOfSquares_;
    UInt_t                   nbOutputLastSumOfSizes_;
    
  };

} // namespace evf


#endif
