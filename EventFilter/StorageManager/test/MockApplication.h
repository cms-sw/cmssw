// -*- c++ -*-
// $Id: MockApplication.h,v 1.7 2011/04/07 09:27:51 mommsen Exp $

#ifndef MOCKAPPLICATION_H
#define MOCKAPPLICATION_H

// xdaq application implementation to be used by unit tests

#include "log4cplus/logger.h"
#include "log4cplus/configurator.h"

#include "toolbox/exception/Handler.h"
#include "toolbox/exception/Processor.h"
#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationContextImpl.h"
#include "xdaq/ApplicationDescriptor.h"
#include "xdaq/ApplicationDescriptorImpl.h"
#include "xdaq/ApplicationStub.h"
#include "xdaq/ApplicationStubImpl.h"
#include "xdaq/ContextDescriptor.h"
#include "xdaq/exception/Exception.h"
#include "xdata/InfoSpace.h"

namespace stor
{
  
  class MockApplicationStub : public xdaq::ApplicationStub
  {

  public:

    MockApplicationStub() :
    logger_( Logger::getRoot() ) //place holder, overwritten below
    {
      log4cplus::BasicConfigurator config;
      config.configure();
      logger_ = Logger::getInstance("main");

      appContext_ = new xdaq::ApplicationContextImpl(logger_);
      //appContext_->init(0, 0);
      
      appDescriptor_ = new xdaq::ApplicationDescriptorImpl
        (
          new xdaq::ContextDescriptor( "none" ),
          "MockApplication", 0, "UnitTests"
        );
      ((xdaq::ApplicationDescriptorImpl*)appDescriptor_)->setInstance(1);
      
      ispace_ = new xdata::InfoSpace("MockApplication");
    }

    virtual ~MockApplicationStub()
    {
      delete ispace_;
    }

    xdaq::ApplicationContext* getContext() { return appContext_; }
    xdaq::ApplicationDescriptor* getDescriptor() { return appDescriptor_; }
    xdata::InfoSpace* getInfoSpace() { return ispace_; }
    Logger& getLogger() { return logger_; }

  private:

    Logger logger_;
    xdaq::ApplicationContextImpl* appContext_;
    xdaq::ApplicationDescriptor* appDescriptor_;
    xdata::InfoSpace* ispace_;

  };
  

  class MockApplicationStubImpl : public xdaq::ApplicationStubImpl
  {

  public:
  
    MockApplicationStubImpl(
      xdaq::ApplicationContext* c, 
      xdaq::ApplicationDescriptor* d, 
      Logger& logger
    )
    : ApplicationStubImpl(c,d,logger)
    {}
  };


  class MockApplication : public xdaq::Application
  {
    
  public:

    MockApplication(MockApplicationStubImpl* s) :
    Application(s)
    {
      toolbox::exception::HandlerSignature* defaultExceptionHandler =
        toolbox::exception::bind (this, &MockApplication::handleException, 
          "MockApplicationStubImpl::handleException");
      toolbox::exception::getProcessor()->setDefaultHandler(defaultExceptionHandler);
    }

    ~MockApplication() {};

    bool handleException(xcept::Exception& ex, void* context)
    {
      return true;
    }
  };


  namespace mockapps
  {
    MockApplication* getMockXdaqApplication()
    {
        MockApplicationStub* stub( new MockApplicationStub() );
        MockApplicationStubImpl* stubImpl( new MockApplicationStubImpl
          (
            stub->getContext(),
            stub->getDescriptor(),
            stub->getLogger()
          )
        );
        MockApplication* app( new MockApplication( stubImpl ) );

        return app;
    }
  } // namespace mockapps
} // namespace stor

#endif // MOCKAPPLICATION_H


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
