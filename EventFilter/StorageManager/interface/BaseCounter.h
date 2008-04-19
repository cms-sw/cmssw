#ifndef STOR_BASE_COUNTER_H
#define STOR_BASE_COUNTER_H

/**
 * This class serves as the base class for the smart counters used as
 * part of storage manager monitoring.  Currently, it simply provides a
 * single method to determine the current time, but there may be more
 * functionality that can be factored out later.
 *
 * $Id: BaseCounter.h,v 1.12 2008/03/03 20:09:36 biery Exp $
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
