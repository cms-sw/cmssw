#ifndef SMPROXY_H
#define SMPROXY_H


#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "xdaq/Application.h"


#include <string>


namespace evf {

  class SMProxy
  {
  public:  
    //
    // construction/destruction
    //
    SMProxy(xdaq::ApplicationDescriptor *fuAppDesc, 
	    xdaq::ApplicationDescriptor *smAppDesc, 
	    xdaq::ApplicationContext    *fuAppContext,
	    toolbox::mem::Pool          *i2oPool);
    virtual ~SMProxy();
    
    
    //
    // member functions
    //
    UInt_t sendInitMessage(UInt_t  fuResourceId,
			   UChar_t*data,
			   UInt_t  dataSize) throw (evf::Exception);
    
    UInt_t sendDataEvent(UInt_t  fuResourceId,
			 UInt_t  runNumber,
			 UInt_t  evtNumber,
			 UInt_t  outModId,
			 UChar_t*data,
			 UInt_t  dataSize) throw (evf::Exception);

    UInt_t sendErrorEvent(UInt_t  fuResourceId,
			  UInt_t  runNumber,
			  UInt_t  evtNumber,
			  UChar_t*data,
			  UInt_t  dataSize) throw (evf::Exception);

    UInt_t sendDqmEvent(UInt_t  fuDqmId,
			UInt_t  runNumber,
			UInt_t  evtAtUpdate,
			UInt_t  folderId,
			UChar_t*data,
			UInt_t  dataSize) throw (evf::Exception);


  private:    
    //
    // private member functions
    //
    MemRef_t* createFragmentChain(UShort_t i2oFunctionCode,
				  UInt_t   headerSize,
				  UChar_t *data,
				  UInt_t   dataSize,
				  UInt_t  &totalSize) throw (evf::Exception);
    
  private:
    //
    // member data
    //
    Logger                       log_;
    xdaq::ApplicationDescriptor *fuAppDesc_;
    xdaq::ApplicationDescriptor *smAppDesc_;
    xdaq::ApplicationContext    *fuAppContext_;
    toolbox::mem::Pool          *i2oPool_;

    UInt_t                       initHeaderSize_;
    UInt_t                       dataHeaderSize_;
    UInt_t                       dqmHeaderSize_;

    std::string                  fuUrl_;
    std::string                  fuClassName_;
    
  }; 

} // namespace evf


#endif
