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

*/


#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "toolbox/mem/MemoryPoolFactory.h"

class SMi2oSender: public xdaq::Application
{
  public:

  //XDAQ_INSTANTIATOR();

  SMi2oSender(xdaq::ApplicationStub * s) throw (xdaq::exception::Exception);

  virtual ~SMi2oSender(){}

  private:

  toolbox::mem::Pool          *pool_;
  vector<xdaq::ApplicationDescriptor*> destinations_;
  xdaq::ApplicationDescriptor* firstDestination_;

};

#endif
