// -*- c++ -*-
// $Id: MockDQMEventProcessorResources.h,v 1.2 2009/06/10 08:15:30 dshpakov Exp $

#ifndef MOCKDQMEVENTPROCESSORRESOURCES_H
#define MOCKDQMEVENTPROCESSORRESOURCES_H

#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"

namespace stor
{

  class MockDQMEventProcessorResources : public DQMEventProcessorResources
  {

  public:

    MockDQMEventProcessorResources() {}

    ~MockDQMEventProcessorResources() {}

    void waitForCompletion() { return; }

    bool requestsOngoing() { return false; }

  };

}

#endif // MOCKDQMEVENTPROCESSORRESOURCES_H


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
