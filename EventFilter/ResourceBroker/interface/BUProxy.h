#ifndef BUPROXY_H
#define BUPROXY_H


#include "EventFilter/ResourceBroker/interface/FUTypes.h"


namespace xdaq {class ApplicationDescriptor; class ApplicationContext; }
namespace toolbox { namespace mem { class Pool; } }


namespace evf {

  class BUProxy
  {
  public:  
    //
    // construction/destruction
    //
    BUProxy (xdaq::ApplicationDescriptor *fuAppDesc, 
	     xdaq::ApplicationDescriptor *buAppDesc, 
	     xdaq::ApplicationContext    *appContext,
	     toolbox::mem::Pool          *i2oPool=0,
	     UInt_t                       dataBufSize=1024);
    
    virtual ~BUProxy();
    
    
    //
    // member functions
    //
    void sendAllocate(const UIntVec_t& fuResourceIds);
    void sendCollect(UInt_t fuResourceId);
    void sendDiscard(UInt_t buResourceId);
    
    
  private:
    //
    // member data
    //
    xdaq::ApplicationDescriptor *fuAppDesc_;
    xdaq::ApplicationDescriptor *buAppDesc_;
    xdaq::ApplicationContext    *appContext_;
    toolbox::mem::Pool          *i2oPool_;
    UInt_t                       dataBufSize_;
    
  }; 

} // namespace evf


#endif
