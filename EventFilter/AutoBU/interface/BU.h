#ifndef AUTOBU_BU_H
#define AUTOBU_BU_H 1


#include "EventFilter/Utilities/interface/StateMachine.h"
#include "EventFilter/Utilities/interface/WebGUI.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "EventFilter/Playback/interface/PlaybackRawDataProvider.h"

#include "xdaq/include/xdaq/Application.h"

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

//#include "extern/i2o/include/i2o/i2oDdmLib.h"


#include <vector>
#include <cmath>
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
    inline void I2O_BU_ALLOCATE_Callback(toolbox::mem::Reference *bufRef);
    inline void I2O_BU_DISCARD_Callback(toolbox::mem::Reference *bufRef);
    
    // xdata::ActionListener callback
    void actionPerformed(xdata::Event& e);

    // Hyper DAQ web interface [see Utilities/WebGUI]
    void webPageRequest(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
  
    // calculate monitoring information in separate thread
    void startMonitoringWorkLoop() throw (evf::Exception);
    bool monitoring(toolbox::task::WorkLoop* wl);
    
    
  private:
    //
    // private member functions
    //
    
    void   exportParameters();
    void   reset();
    double deltaT(const struct timeval *start,const struct timeval *end);
    
    // fed buffers for one superfragment
    void initFedBuffers(unsigned int nFed);
    void fillFedBuffers(unsigned int iSuperFrag,FEDRawDataCollection* event);
    void clearFedBuffers();
    
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
    void dumpFrame(unsigned char* data,unsigned int len);
  
  
  private:
    //
    // member data
    //
    Logger                    log_;
    StateMachine              fsm_;
    WebGUI*                   gui_;
    
    // workloop / action signature for monitoring
    toolbox::task::WorkLoop *wlMonitoring_;      
    toolbox::task::ActionSignature *asMonitoring_;
    
    std::string               sourceId_;
    
    unsigned int             *fedN_;       // nFED/SF,    [iSF ]
    unsigned char           **fedData_;    // current SF, [iFED][pos]
    unsigned int             *fedSize_;    // current SF, [iFED]
    unsigned int            **fedId_;      // fedid,      [iSF ][iFED]
    
    // parameters and counters to be exported

    // monitored parameters
    xdata::String             url_;
    xdata::String             class_;
    xdata::UnsignedInteger32  instance_;
    xdata::String             hostname_;
    xdata::UnsignedInteger32  runNumber_;
    xdata::Double             memUsedInMB_;

    xdata::Double             deltaT_;
    xdata::UnsignedInteger32  deltaN_;
    xdata::Double             deltaSumOfSquares_;
    xdata::UnsignedInteger32  deltaSumOfSizes_;

    xdata::Double             throughput_;
    xdata::Double             average_;
    xdata::Double             rate_;
    xdata::Double             rms_;
    
    // monitored counters
    xdata::UnsignedInteger32  nbEventsInBU_;
    xdata::UnsignedInteger32  nbEventsBuilt_;
    xdata::UnsignedInteger32  nbEventsDiscarded_;
    
    // standard parameters
    xdata::String             mode_;
    xdata::UnsignedInteger32  dataBufSize_;
    xdata::UnsignedInteger32  nSuperFrag_;
    xdata::UnsignedInteger32  fedSizeMax_;
    xdata::UnsignedInteger32  fedSizeMean_;
    xdata::UnsignedInteger32  fedSizeWidth_;
    xdata::Boolean            useFixedFedSize_;
    xdata::UnsignedInteger32  monSleepSec_;
    
    // debug counters
    xdata::UnsignedInteger32  nbPostFrame_;
    xdata::UnsignedInteger32  nbPostFrameFailed_;
    
    // monitoring helpers
    struct timeval            monStartTime_;
    unsigned int              monLastN_;
    uint64_t                  monLastSumOfSquares_;
    unsigned int              monLastSumOfSizes_;
    uint64_t                  sumOfSquares_;
    unsigned int              sumOfSizes_;
    

    // memory pool for i20 communication
    toolbox::mem::Pool*       i2oPool_;

    // synchronization
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
