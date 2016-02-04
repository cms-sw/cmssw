#ifndef BUPROXY_H
#define BUPROXY_H


#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "xdaq/Application.h"


namespace evf {

  class BUProxy
  {
  public:  
    //
    // construction/destruction
    //
    BUProxy(xdaq::ApplicationDescriptor *fuAppDesc, 
	    xdaq::ApplicationDescriptor *buAppDesc, 
	    xdaq::ApplicationContext    *fuAppContext,
	    toolbox::mem::Pool          *i2oPool);
    virtual ~BUProxy();
    
    
    //
    // member functions
    //
    void sendAllocate(const UIntVec_t& fuResourceIds) throw (evf::Exception);
    void sendDiscard(UInt_t buResourceId) throw (evf::Exception);
    
    
  private:
    //
    // member data
    //
    xdaq::ApplicationDescriptor *fuAppDesc_;
    xdaq::ApplicationDescriptor *buAppDesc_;
    xdaq::ApplicationContext    *fuAppContext_;
    toolbox::mem::Pool          *i2oPool_;
  
  }; 

} // namespace evf


#endif
