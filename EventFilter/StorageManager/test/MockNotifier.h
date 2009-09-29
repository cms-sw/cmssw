// -*- c++ -*-
// $Id: MockNotifier.h,v 1.5 2009/07/03 18:39:40 mommsen Exp $

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

  private:

    MockApplication* _app;

  };

}

#endif // MOCKNOTIFIER_H
