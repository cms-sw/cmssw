#ifndef _SMI2OSENDER_H_
#define _SMI2OSENDER_H_

/*
   Author: Harry Cheung, FNAL

   Description:
     Used for FU I2O frame output module.
     See CMS EventFilter wiki page for further notes.

   Modification:
     version 1.1 2005/11/23
       Initial implementation, only creates pool and destination.
       Uses a global pointer. Needs changes for production version.
     version 1.2 2005/12/15
       Changed to using a committed heap memory pool allocator and
       a way to set its size.
       Added default home page to show statistics.

*/


#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "xdata/UnsignedLong.h"

#include "xgi/include/xgi/Input.h"
#include "xgi/include/xgi/Output.h"
#include "xgi/include/xgi/exception/Exception.h"
#include "EventFilter/Utilities/interface/Css.h"

class SMi2oSender: public xdaq::Application
{
  public:

  //XDAQ_INSTANTIATOR();

  SMi2oSender(xdaq::ApplicationStub * s) throw (xdaq::exception::Exception);

  virtual ~SMi2oSender(){}

  void defaultWebPage
    (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);

  private:

  void css(xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception)
    {css_.css(in,out);}

  toolbox::mem::Pool          *pool_;
  vector<xdaq::ApplicationDescriptor*> destinations_;
  xdaq::ApplicationDescriptor* firstDestination_;
  xdata::UnsignedLong          committedpoolsize_;
  evf::Css css_;

};

#endif
