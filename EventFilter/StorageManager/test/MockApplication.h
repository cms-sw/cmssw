// -*- c++ -*-
// $Id: MockApplication.h,v 1.4 2009/07/03 18:44:35 mommsen Exp $

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
    _logger( Logger::getRoot() ) //place holder, overwritten below
    {
      log4cplus::BasicConfigurator config;
      config.configure();
      _logger = Logger::getInstance("main");

      _appContext = new xdaq::ApplicationContextImpl(_logger);
      //_appContext->init(0, 0);
      
      _appDescriptor = new xdaq::ApplicationDescriptorImpl
        (
          new xdaq::ContextDescriptor( "none" ),
          "MockApplication", 0, "UnitTests"
        );
      
      _ispace = new xdata::InfoSpace("MockApplication");
    }

    virtual ~MockApplicationStub()
    {
      delete _ispace;
    }

    xdaq::ApplicationContext* getContext() { return _appContext; }
    xdaq::ApplicationDescriptor* getDescriptor() { return _appDescriptor; }
    xdata::InfoSpace* getInfoSpace() { return _ispace; }
    Logger& getLogger() { return _logger; }

  private:

    Logger _logger;
    xdaq::ApplicationContextImpl* _appContext;
    xdaq::ApplicationDescriptor* _appDescriptor;
    xdata::InfoSpace* _ispace;

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
