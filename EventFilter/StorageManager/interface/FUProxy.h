#ifndef StorageManager_FUProxy_h
#define StorageManager_FUProxy_h

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "xdaq/Application.h"

#include <string>

////////////////////////////////////////////////////////////////////////////////
namespace stor 
{

  /**
   * Send back discards to the FU resource borker.
   *
   * Following the example in EventFilter/ResourceBroker for BUProxy and SMProxy.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:21 $
   */

  class FUProxy
    {     
    public:
      FUProxy(xdaq::ApplicationDescriptor *,
	      xdaq::ApplicationDescriptor *,
	      xdaq::ApplicationContext    *,
	      toolbox::mem::Pool          *);

      virtual ~FUProxy();
      
      int sendDataDiscard(int);
      int sendDQMDiscard(int);

    private:
      int sendDiscard(int, int)   throw (evf::Exception);

      xdaq::ApplicationDescriptor *smAppDesc_;
      xdaq::ApplicationDescriptor *fuAppDesc_;
      xdaq::ApplicationContext    *smAppContext_;
      toolbox::mem::Pool          *i2oPool_;
      
    };

} // namespace stor

#endif // StorageManager_FUProxy_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
