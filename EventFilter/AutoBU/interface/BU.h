#ifndef AUTOBU_BU_H
#define AUTOBU_BU_H 1


#include "EventFilter/Utilities/interface/EPStateMachine.h"
#include "EventFilter/Utilities/interface/WebGUI.h"

#include "EventFilter/Playback/interface/PlaybackRawDataProvider.h"

#include "xdaq/include/xdaq/Application.h"

#include "toolbox/include/toolbox/task/TimerFactory.h"
#include "toolbox/include/Task.h"
#include "toolbox/include/toolbox/mem/HeapAllocator.h"
#include "toolbox/include/toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/include/toolbox/net/URN.h"
#include "toolbox/include/toolbox/fsm/exception/Exception.h"
#include "toolbox/include/BSem.h"

#include "xdata/include/xdata/InfoSpace.h"
#include "xdata/include/xdata/UnsignedInteger32.h"
#include "xdata/include/xdata/Double.h"
#include "xdata/include/xdata/Boolean.h"
#include "xdata/include/xdata/String.h"

#include "interface/evb/include/i2oEVBMsgs.h"
#include "interface/shared/include/i2oXFunctionCodes.h"

#include "interface/shared/include/frl_header.h"
#include "interface/shared/include/fed_header.h"
#include "interface/shared/include/fed_trailer.h"

#include "i2o/include/i2o/Method.h"
#include "i2o/utils/include/i2o/utils/AddressMap.h"

#include "CLHEP/Random/RandGauss.h"

#include "extern/i2o/include/i2o/i2oDdmLib.h"


#include <vector>
#include <cmath>


namespace evf {


  class BU : public xdaq::Application,
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
    BU(xdaq::ApplicationStub *s);
    virtual ~BU();
  
  
    //
    // public member functions
    //

    // toolbox::task::TimerListener callback
    void timeExpired(toolbox::task::TimerEvent& e);
    void initTimer();
    void startTimer();
    void stopTimer();

    // xdata::ActionListener callback
    void actionPerformed(xdata::Event& e);

    // finite state machine callbacks
    void configureAction(toolbox::Event::Reference e)
      throw (toolbox::fsm::exception::Exception);
    void enableAction(toolbox::Event::Reference e)
      throw (toolbox::fsm::exception::Exception);
    void suspendAction(toolbox::Event::Reference e)
      throw (toolbox::fsm::exception::Exception);
    void resumeAction(toolbox::Event::Reference e)
      throw (toolbox::fsm::exception::Exception);
    void haltAction(toolbox::Event::Reference e)
      throw (toolbox::fsm::exception::Exception);
    void nullAction(toolbox::Event::Reference e)
      throw (toolbox::fsm::exception::Exception);
    
    xoap::MessageReference fireEvent(xoap::MessageReference msg)
      throw (xoap::exception::Exception);

    // Hyper DAQ web interface [see Utilities/WebGUI]
    void webPageRequest(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    
    // i2o callbacks
    inline void I2O_BU_ALLOCATE_Callback(toolbox::mem::Reference *bufRef);
    inline void I2O_BU_COLLECT_Callback(toolbox::mem::Reference *bufRef);
    inline void I2O_BU_DISCARD_Callback(toolbox::mem::Reference *bufRef);
    
    // export parameters to info space(s)
    void exportParameters();
    
    
  private:
    //
    // private member functions
    //
    
    // initialize/clean up internal FED data buffers
    void initFedBuffers(unsigned int nFed);
    void clearFedBuffers();

    // generate FEDs with random payload (including headers/trailers, not yet valid!)
    void generateRndmFEDs(unsigned int iSuperFrag);
  
    //estimate number of blocks needed for a superfragment
    int  estimateNBlocks(unsigned int iSuperFrag,unsigned int fullBlockPayload);
    
    // create a supefragment
    inline
    toolbox::mem::Reference *createSuperFrag(const I2O_TID& fuTid,
					     const U32&     fuTransaction,
					     const U32&     trigNo,
					     const U32&     iSuperFrag,
					     const U32&     nSuperFrag);
    
    // synchronization operations
    void lock();
    void unlock();

    // debug functionality
    void debug(toolbox::mem::Reference* ref);
    int  check_event_data(unsigned long* blocks_adrs,int nmb_blocks);
    void dumpFrame(unsigned char* data,unsigned int len);
  
  
  private:
    //
    // member data
    //
    Logger                    log_;
    EPStateMachine*           fsm_;
    WebGUI*                   gui_;
    
    std::string               sourceId_;

    unsigned int             *fedN_;       // nFED/SF, [iSF]
    unsigned char           **fedData_;    // current SF, [iFED][pos]
    unsigned int             *fedSize_;    // current SF, [iFED]

    unsigned int            **fedId_;      // fedid, [iSF][iFED] (RANDOM only)
    
    // parameters and counters to be exported
    xdata::String             url_;
    xdata::String             class_;
    xdata::UnsignedInteger32  instance_;
    xdata::String             hostname_;
    xdata::UnsignedInteger32  runNumber_;
    xdata::Double             nbMBTot_;
    xdata::Double             nbMBPerSec_;
    xdata::Double             nbMBPerSecMin_;
    xdata::Double             nbMBPerSecMax_;
    xdata::Double             nbMBPerSecAvg_;
    xdata::Double             memUsedInMB_;
    xdata::UnsignedInteger32  nbEventsInBU_;
    xdata::Double             deltaT_;
    xdata::UnsignedInteger32  deltaN_;
    xdata::UnsignedInteger32  deltaSumOfSquares_;
    xdata::UnsignedInteger32  deltaSumOfSizes_;
    
    xdata::UnsignedInteger32  nbEvents_;
    xdata::UnsignedInteger32  nbEventsPerSec_;
    xdata::UnsignedInteger32  nbEventsPerSecMin_;
    xdata::UnsignedInteger32  nbEventsPerSecMax_;
    xdata::UnsignedInteger32  nbEventsPerSecAvg_;
    xdata::UnsignedInteger32  nbDiscardedEvents_;
    
    xdata::String             mode_;
    xdata::Boolean            debug_;
    xdata::UnsignedInteger32  dataBufSize_;
    xdata::UnsignedInteger32  nSuperFrag_;
    xdata::UnsignedInteger32  fedSizeMax_;
    xdata::UnsignedInteger32  fedSizeMean_;
    xdata::UnsignedInteger32  fedSizeWidth_;
    xdata::Boolean            useFixedFedSize_;
    
    xdata::UnsignedInteger32  nbPostFrame_;
    xdata::UnsignedInteger32  nbPostFrameFailed_;
    
    // internal parameters and counters (not to be exported)
    unsigned int              nbMeasurements_;
    unsigned int              nbEventsLast_;
    unsigned int              nbBytes_;
    unsigned int              sumOfSquares_;
    unsigned int              sumOfSizes_;
    
    // memory pool for i20 communication
    toolbox::mem::Pool*       i2oPool_;

    // binary semaphore
    BSem                      lock_;
  
  
    //
    // static member data
    //
    static const int frlHeaderSize_ =sizeof(frlh_t);
    static const int fedHeaderSize_ =sizeof(fedh_t);
    static const int fedTrailerSize_=sizeof(fedt_t);
  
  }; // class BU


} // namespace evf


#endif
