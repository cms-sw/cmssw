// -*- c++ -*-
// $Id: MockApplication.h,v 1.2 2009/06/10 08:15:30 dshpakov Exp $

#ifndef MOCKAPPLICATION_H
#define MOCKAPPLICATION_H

// xdaq application implementation to be used by unit tests

#include "log4cplus/logger.h"
#include "log4cplus/configurator.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationContextImpl.h"
#include "xdaq/ApplicationDescriptor.h"
#include "xdaq/ApplicationDescriptorImpl.h"
#include "xdaq/ApplicationStub.h"
#include "xdaq/ContextDescriptor.h"
#include "xdaq/exception/Exception.h"
#include "xdata/InfoSpace.h"

namespace stor
{
  
  class MockApplicationStub : public xdaq::ApplicationStub
  {

  public:

    MockApplicationStub() :
    _logger( Logger::getRoot() ), //place holder, overwritten below
    _appContext( new xdaq::ApplicationContextImpl(_logger) ),
    _appDescriptor( new xdaq::ApplicationDescriptorImpl(
        new xdaq::ContextDescriptor( "none" ),
        "MockApplication", 0, "UnitTests"
      )
    )
    {
      log4cplus::BasicConfigurator config;
      config.configure();
      _logger = Logger::getInstance("main");
      LOG4CPLUS_FATAL(_logger, "test");

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
    xdaq::ApplicationContext* _appContext;
    xdaq::ApplicationDescriptor* _appDescriptor;
    xdata::InfoSpace* _ispace;

  };
  

  class MockApplication : public xdaq::Application
  {
    
  public:
    
    MockApplication(MockApplicationStub* s) :
    Application(s),
    _stub(s)
    {}

    ~MockApplication() {};

    void notifyQualified(std::string severity, xcept::Exception&) {}
    Logger& getApplicationLogger() { return _stub->getLogger(); }

  private:

    MockApplicationStub* _stub;
  };

}

#endif // MOCKAPPLICATION_H


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
