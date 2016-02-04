#ifndef STOR_BASE_COUNTER_H
#define STOR_BASE_COUNTER_H

/**
 * This class serves as the base class for the smart counters used as
 * part of storage manager monitoring.  Currently, it simply provides a
 * single method to determine the current time, but there may be more
 * functionality that can be factored out later.
 *
 * $Id: BaseCounter.h,v 1.1 2008/04/14 15:42:28 biery Exp $
 */

#include <iostream>

namespace stor
{
  class BaseCounter
  {

   public:

    static double getCurrentTime();

  };
}

#endif
