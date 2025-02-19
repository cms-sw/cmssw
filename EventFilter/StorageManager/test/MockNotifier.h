// -*- c++ -*-
// $Id: MockNotifier.h,v 1.8 2011/03/07 15:31:32 mommsen Exp $

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
      app_( app )
    {}
    
    ~MockNotifier() {}

    void reportNewState( const std::string& stateName ) {}

  private:

    xdaq::Application* app_;

  };

}

#endif // MOCKNOTIFIER_H


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
