/**
 * $Id: BaseCounter.cc,v 1.19 2008/03/03 20:09:37 biery Exp $
 */

#include "EventFilter/StorageManager/interface/BaseCounter.h"
#include <sys/time.h>
#include <iostream>

using namespace stor;

/**
 * Returns the current time as a double.  The value corresponds to the
 * number of seconds since the epoch (including a fractional part good to
 * the microsecond level).  A negative value indicates that an error
 * occurred when fetching the time from the operating system.
 */
double BaseCounter::getCurrentTime()
{
  double now = -1.0;
  struct timeval timeStruct;
  int status = gettimeofday(&timeStruct, 0);
  if (status == 0) {
    now = ((double) timeStruct.tv_sec) +
      (((double) timeStruct.tv_usec) / 1000000.0);
  }
  return now;
}
