// -*- c++ -*-
// $Id$

#ifndef MOCKDISKWRITERRESOURCES_H
#define MOCKDISKWRITERRESOURCES_H

#include "EventFilter/StorageManager/interface/DiskWriterResources.h"

namespace stor
{

  class MockDiskWriterResources : public DiskWriterResources
  {

  public:

    MockDiskWriterResources() {}

    ~MockDiskWriterResources() {}

    void waitForStreamChange() { return; }

    bool streamChangeOngoing() { return false; }

  };

}

#endif // MOCKDISKWRITERRESOURCES_H


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
