// -*- c++ -*-
// $Id: MockNotifier.h,v 1.6 2009/09/29 08:00:41 mommsen Exp $

#ifndef MOCKNOTIFIER_H
#define MOCKNOTIFIER_H

// Notifier implementation to be used by the state machine unit test

#include "EventFilter/StorageManager/interface/Notifier.h"

#include "xdaq/Application.h"


namespace stor
{

  class MockNotifier: public Notifier
  {

  public:

    MockNotifier( xdaq::Application* app ):
      _app( app )
    {}
    
    ~MockNotifier() {}

    void reportNewState( const std::string& stateName ) {}

  private:

    xdaq::Application* _app;

  };

}

#endif // MOCKNOTIFIER_H


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
