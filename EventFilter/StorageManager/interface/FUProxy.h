#if !defined(STOR_FUPROXY_H)
#define STOR_FUPROXY_H

////////////////////////////////////////////////////////////////////////////////
// Created by Markus Klute on 2007 Mar 22.
// $Id: FUProxy.h,v 1.1 2007/03/29 07:17:46 klute Exp $
////////////////////////////////////////////////////////////////////////////////
// send back discards to filter units
//
// following the example in EventFilter/ResourceBroker for BUProxy and SMProxy
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "xdaq/Application.h"

#include <string>

////////////////////////////////////////////////////////////////////////////////
namespace stor 
{
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

#endif
