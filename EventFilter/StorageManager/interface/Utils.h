// $Id: Utils.h,v 1.2 2009/06/10 08:15:24 dshpakov Exp $

#ifndef StorageManager_Utils_h
#define StorageManager_Utils_h

#include <string>

#include "xcept/Exception.h"


namespace xdaq
{
  class Application;
  class ApplicationDescriptor;
}


namespace stor {

  namespace utils {

    /**
     * Collection of utility functions used in the storage manager
     *
     * $Author: dshpakov $
     * $Revision: 1.2 $
     * $Date: 2009/06/10 08:15:24 $
     */

    /**
       time_point_t is used to represent a specific point in time,
       measured by some specific clock. We rely on the "system
       clock". The value is represented by the number of seconds
       (including a fractional part that depends on the resolution of
       the system clock) since the beginning of the "epoch" (as defined
       by the system clock).
    */
    typedef double time_point_t;

    /**
       durtion_t is used to represent a duration (the "distance" between
       two points in time). The value is represented as a number of
       seconds (including a fractional part).
    */
    typedef double duration_t;

    /**
       Returns the current point in time. A negative value indicates
       that an error occurred when fetching the time from the operating
       system.
    */
    time_point_t getCurrentTime();

    /**
       Sleep for at least the given duration. Note that the underlying
       system will round the interval up to an integer multiple of the
       system's sleep resolution. (The underlying implementation
       relies upon nanosleep, so see documentation for nanosleep for
       details). Negative intervals are not allowed, and result in an
       error (returning -1, and no sleeping).
     */
    int sleep(duration_t interval);

    /**
       Converts a time_point_t into a string.
       Note: the string formatting is used by the file summary catalog and
       may or may not depend on the actual formatting
    */
    std::string timeStamp(time_point_t);

    /**
       Returns an identifier string composed of class name and instance
    */
    std::string getIdentifier(xdaq::ApplicationDescriptor*);
    
    /**
       Throws a stor::exception::NoSuchDirectory when the directory does not exist
     */
    void checkDirectory(const std::string&);

    /**
       Raises a sentinel alarm
    */
    void raiseAlarm
    (
      const std::string name,
      const std::string level,
      xcept::Exception&,
      xdaq::Application*
    );

    /**
       Revokes s sentinel alarm
    */
    void revokeAlarm(const std::string name, xdaq::Application*);
 

  } // namespace utils
  
} // namespace stor

#endif // StorageManager_Utils_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
