// -*- c++ -*-
// $Id: MockDiskWriterResources.h,v 1.2 2009/06/10 08:15:30 dshpakov Exp $

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
