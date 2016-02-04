// $Id: FUProxy.h,v 1.6 2009/07/20 13:06:10 mommsen Exp $
/// @file: FUProxy.h 

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
   * $Author: mommsen $
   * $Revision: 1.6 $
   * $Date: 2009/07/20 13:06:10 $
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
