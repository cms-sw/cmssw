#ifndef _SMI2OSENDER_H_
#define _SMI2OSENDER_H_

/*
   Description:
     XDAQ application used for FU I2O frame output module.
     See CMS EvF Storage Manager wiki page for further notes.

   $Id$
*/

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "xdata/UnsignedInteger32.h"

#include "xgi/include/xgi/Input.h"
#include "xgi/include/xgi/Output.h"
#include "xgi/include/xgi/exception/Exception.h"
#include "EventFilter/Utilities/interface/Css.h"

class SMi2oSender: public xdaq::Application
{
  public:

  //XDAQ_INSTANTIATOR(); // This is the standard one that we may want to go back to

  SMi2oSender(xdaq::ApplicationStub * s) throw (xdaq::exception::Exception);

  virtual ~SMi2oSender(){}

  void defaultWebPage
    (xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception);

  void setDestinations();

  static toolbox::mem::Reference
      *createI2OFragmentChain(char *rawData,
                              unsigned int dataSize,
                              toolbox::mem::Pool *fragmentPool,
                              unsigned int maxFragmentSize,
                              unsigned int trueHeaderSize,
                              unsigned short functionCode,
                              xdaq::Application *sourceApp,
                              xdaq::ApplicationDescriptor *destAppDesc,
                              unsigned int& numBytesSend);

  static void debugFragmentChain(toolbox::mem::Reference *head);

  static void waitForPoolSpaceIfNeeded(toolbox::mem::Pool *targetPool);

  private:

  void css(xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception)
    {css_.css(in,out);}

  toolbox::mem::Pool          *pool_;
  set<xdaq::ApplicationDescriptor*> destinations_;
  xdaq::ApplicationDescriptor* firstDestination_;
  xdata::UnsignedInteger32          committedpoolsize_;
  xdata::UnsignedInteger32     primarysm_;
  evf::Css css_;

};

#endif
