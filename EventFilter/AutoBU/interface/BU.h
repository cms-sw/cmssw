#ifndef AUTOBU_BU_H
#define AUTOBU_BU_H 1


#include "EventFilter/Utilities/interface/EPStateMachine.h"
#include "EventFilter/Utilities/interface/WebGUI.h"

#include "EventFilter/Playback/interface/PlaybackRawDataProvider.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

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
    // typedefs
    //
    typedef std::vector<unsigned char*> UCharVec_t;
    typedef std::vector<unsigned int>   UIntVec_t;
  
  
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

    // xdata::ActionListener callback(s)
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

    // Hyper DAQ web interface [see Utilities/WebGUI!]
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
    void initFedBuffers(unsigned int fedN);
    void clearFedBuffers();

    // generate FEDs with random payload (including headers/trailers, not yet valid!)
    void generateRndmFEDs();
  
    //estimate number of blocks needed for a superfragment
    int estimateNBlocks(size_t fullBlockPayload);
  
    // create a supefragment
    toolbox::mem::Reference *createSuperFrag(const I2O_TID& fuTid,
					     const U32&     fuTransaction,
					     const U32&     trigNo,
					     const U32&     iSuperFrag,
					     const U32&     nSuperFrag);
  
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
    
    std::string               xmlClass_;
    unsigned int              instance_;
    std::string               sourceId_;

    unsigned int              fedN_;
    unsigned char           **fedData_;
    unsigned int             *fedSize_;
    unsigned int              fedSizeMax_;
    
    //UCharVec_t                fedData_;
    //UIntVec_t                 fedSize_;

    // parameters and counters to be exported
    xdata::String             mode_;
    xdata::Boolean            debug_;
    xdata::UnsignedInteger32  dataBufSize_;

    xdata::UnsignedInteger32  nSuperFrag_;
    xdata::UnsignedInteger32  fedSizeMean_;
    xdata::UnsignedInteger32  fedSizeWidth_;
    xdata::Boolean            useFixedFedSize_;
    
    xdata::UnsignedInteger32  nbEvents_;
    xdata::UnsignedInteger32  nbEventsPerSec_;
    xdata::UnsignedInteger32  nbDiscardedEvents_;
    xdata::Double             nbMBPerSec_;
    
    // internal parameters and counters (not to be exported)
    xdata::UnsignedInteger32  nbEventsLast_; // for nbEventsPerSec measurement
    xdata::UnsignedInteger32  nbBytes_;      // for MB/s measurement
    
    // memory pool for i20 communication
    toolbox::mem::Pool*       i2oPool_;

    // binary semaphore
    BSem                      bSem_;
  
  
    //
    // static member data
    //
    static const int frlHeaderSize_ =sizeof(frlh_t);
    static const int fedHeaderSize_ =sizeof(fedh_t);
    static const int fedTrailerSize_=sizeof(fedt_t);
  
  }; // class BU


} // namespace evf


#endif
