// -*- c++ -*-
// $Id: MockNotifier.h,v 1.4 2009/07/02 12:55:28 dshpakov Exp $

#ifndef MOCKNOTIFIER_H
#define MOCKNOTIFIER_H

// Notifier implementation to be used by the state machine unit test

#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/test/MockApplication.h"

#include "xdaq/Application.h"


namespace stor
{

  class MockNotifier: public Notifier
  {

  public:

    MockNotifier( MockApplication* app ):
      _app( app )
    {}
    
    ~MockNotifier() {}

    void reportNewState( const std::string& stateName ) {}
    Logger& getLogger() { return _app->getApplicationLogger(); }
    void tellSentinel( const std::string& level, xcept::Exception& e ) {}

  private:

    MockApplication* _app;

    unsigned long instanceNumber() const { return 0; }

  };

}

#endif // MOCKNOTIFIER_H
