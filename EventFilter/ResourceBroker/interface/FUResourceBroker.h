#ifndef FURESOURCEBROKER_H
#define FURESOURCEBROKER_H 1


#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/ResourceBroker/interface/FUResourceTable.h"

#include "EventFilter/Utilities/interface/StateMachine.h"
#include "EventFilter/Utilities/interface/WebGUI.h"

#include "xdaq/include/xdaq/Application.h"
#include "xdaq/NamespaceURI.h"

#include "xdata/include/xdata/InfoSpace.h"
#include "xdata/include/xdata/String.h"
#include "xdata/include/xdata/Boolean.h"
#include "xdata/include/xdata/UnsignedInteger32.h"
#include "xdata/include/xdata/Double.h"

#include "toolbox/include/toolbox/task/TimerFactory.h"
#include "toolbox/include/Task.h"
#include "toolbox/include/toolbox/mem/Reference.h"
#include "toolbox/include/toolbox/fsm/exception/Exception.h"
#include "toolbox/include/BSem.h"

#include "interface/shared/include/frl_header.h"
#include "interface/shared/include/fed_header.h"
#include "interface/shared/include/fed_trailer.h"

#include <vector>
#include <string>
#include <semaphore.h>


namespace evf {

  class BUProxy;
  class SMProxy;
  
  class FUResourceBroker : public xdaq::Application,
			   public toolbox::task::TimerListener,
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
    
    // toolbox::task::TimerListener callback, and init/start/stop the corresp. timer
    void timeExpired(toolbox::task::TimerEvent& e);
    void initTimer();
    void startTimer();
    void stopTimer();

    // xdata::ActionListener callback(s)
    void actionPerformed(xdata::Event& e);
    
    
  private:
    //
    // private member functions
    //
    void exportParameters();
    void reset();
    
    
  private:    
    //
    // member data
    //
    
    // finite state machine
    evf::StateMachine        fsm_;
    
    // application identifier
    std::string              sourceId_;
    
    // binary semaphore
    BSem                     lock_;
    
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
    
    // monitored parameters 
    xdata::String            url_;
    xdata::String            class_;
    xdata::UnsignedInteger32 instance_;
    xdata::UnsignedInteger32 runNumber_;
    xdata::UnsignedInteger32 nbShmClients_;
    
    xdata::Double            acceptRate_;
    xdata::Double            nbMBInput_;
    xdata::Double            nbMBInputPerSec_;
    xdata::Double            nbMBInputPerSecMin_;
    xdata::Double            nbMBInputPerSecMax_;
    xdata::Double            nbMBInputPerSecAvg_;
    xdata::Double            nbMBOutput_;
    xdata::Double            nbMBOutputPerSec_;
    xdata::Double            nbMBOutputPerSecMin_;
    xdata::Double            nbMBOutputPerSecMax_;
    xdata::Double            nbMBOutputPerSecAvg_;
    
    // monitored counters
    xdata::UnsignedInteger32 nbInputEvents_;
    xdata::UnsignedInteger32 nbInputEventsPerSec_;
    xdata::UnsignedInteger32 nbInputEventsPerSecMin_;
    xdata::UnsignedInteger32 nbInputEventsPerSecMax_;
    xdata::UnsignedInteger32 nbInputEventsPerSecAvg_;
    xdata::UnsignedInteger32 nbOutputEvents_;
    xdata::UnsignedInteger32 nbOutputEventsPerSec_;
    xdata::UnsignedInteger32 nbOutputEventsPerSecMin_;
    xdata::UnsignedInteger32 nbOutputEventsPerSecMax_;
    xdata::UnsignedInteger32 nbOutputEventsPerSecAvg_;
    
    xdata::UnsignedInteger32 nbAllocatedEvents_;
    xdata::UnsignedInteger32 nbPendingRequests_;
    xdata::UnsignedInteger32 nbProcessedEvents_;
    xdata::UnsignedInteger32 nbAcceptedEvents_;
    xdata::UnsignedInteger32 nbDiscardedEvents_;

    xdata::UnsignedInteger32 nbLostEvents_;
    xdata::UnsignedInteger32 nbDataErrors_;
    xdata::UnsignedInteger32 nbCrcErrors_;
    
    // standard parameters
    xdata::Boolean           segmentationMode_;
    xdata::UnsignedInteger32 nbRawCells_;
    xdata::UnsignedInteger32 nbRecoCells_;
    xdata::UnsignedInteger32 nbDqmCells_;
    xdata::UnsignedInteger32 rawCellSize_;
    xdata::UnsignedInteger32 recoCellSize_;
    xdata::UnsignedInteger32 dqmCellSize_;

    xdata::Boolean           doDropEvents_;
    xdata::Boolean           doFedIdCheck_;
    xdata::UnsignedInteger32 doCrcCheck_;

    xdata::String            buClassName_;
    xdata::UnsignedInteger32 buInstance_;

    xdata::String            smClassName_;
    xdata::UnsignedInteger32 smInstance_;
    
    // debug parameters
    xdata::UnsignedInteger32 nbAllocateSent_;
    xdata::UnsignedInteger32 nbTakeReceived_;
    xdata::UnsignedInteger32 nbDataDiscardReceived_;
    xdata::UnsignedInteger32 nbDqmDiscardReceived_;
    
    // internal parameters, not exported
    unsigned int             nbMeasurements_;
    unsigned int             nbInputEventsLast_;
    unsigned int             nbOutputEventsLast_;
    
  };

} // namespace evf


#endif
