// -*- c++ -*-
// $Id$

#ifndef MOCKNOTIFIER_H
#define MOCKNOTIFIER_H

// Notifier implementation to be used by the state machine unit test

#include "EventFilter/StorageManager/interface/Notifier.h"

namespace stor
{

  class MockNotifier: public Notifier
  {

  public:

    MockNotifier() {}
    ~MockNotifier() {}

    void reportNewState( const std::string& stateName ) {}

  };

}

#endif // MOCKNOTIFIER_H
