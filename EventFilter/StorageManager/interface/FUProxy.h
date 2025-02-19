// $Id: FUProxy.h,v 1.8 2011/04/07 08:02:03 mommsen Exp $
/// @file: FUProxy.h 

#ifndef EventFilter_StorageManager_FUProxy_h
#define EventFilter_StorageManager_FUProxy_h

#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationDescriptor.h"
#include "toolbox/mem/Pool.h"


namespace stor 
{
  
  /**
   * Send back discards to the FU resource borker.
   *
   * Following the example in EventFilter/ResourceBroker for BUProxy and SMProxy.
   *
   * $Author: mommsen $
   * $Revision: 1.8 $
   * $Date: 2011/04/07 08:02:03 $
   */
  
  class FUProxy
  {     
  public:
    FUProxy
    (
      xdaq::ApplicationDescriptor* smAppDesc,
      xdaq::ApplicationDescriptor* fuAppDesc,
      xdaq::ApplicationContext* smAppContext,
      toolbox::mem::Pool* msgPool
    );
      
    void sendDataDiscard(const int& rbBufferId);
    void sendDQMDiscard(const int& rbBufferId);
    
  private:
    void sendDiscardMsg
    (
      const int& rbBufferId,
      const int& msgId,
      const size_t& msgSize
    );
    
    xdaq::ApplicationDescriptor* smAppDesc_;
    xdaq::ApplicationDescriptor* fuAppDesc_;
    xdaq::ApplicationContext* smAppContext_;
    toolbox::mem::Pool* msgPool_;
  };
  
} // namespace stor

#endif // EventFilter_StorageManager_FUProxy_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
