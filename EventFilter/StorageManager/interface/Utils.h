// $Id: Utils.h,v 1.15 2011/03/31 13:04:20 mommsen Exp $
/// @file: Utils.h 

#ifndef EventFilter_StorageManager_Utils_h
#define EventFilter_StorageManager_Utils_h

#include <string>
#include <vector>

#include "xcept/Exception.h"
#include "xdaq/ApplicationDescriptor.h"
#include "xdata/String.h"
#include "xdata/Vector.h"

#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/posix_time/posix_time_types.hpp"
#include "boost/thread/thread.hpp"


namespace stor {

  namespace utils {

    /**
     * Collection of utility functions used in the storage manager
     *
     * $Author: mommsen $
     * $Revision: 1.15 $
     * $Date: 2011/03/31 13:04:20 $
     */

    /**
       TimePoint_t is used to represent a specific point in time
    */
    typedef boost::posix_time::ptime TimePoint_t;

    /**
       durtion_t is used to represent a duration (the "distance" between
       two points in time).
    */
    typedef boost::posix_time::time_duration Duration_t;

    /**
       Convert a fractional second count into a boost::posix_time::time_duration
       type with a resolution of milliseconds.
    */
    Duration_t secondsToDuration(double const& seconds);

    /**
       Convert a boost::posix_time::time_duration into fractional seconds with
       a resolution of milliseconds.
    */
    double durationToSeconds(Duration_t const&);

    /**
       Return the number of seconds since the unix epoch 1-1-1970
    */
    long secondsSinceEpoch(TimePoint_t const&);

    /**
       Returns the current point in time.
    */
    TimePoint_t getCurrentTime();

    /**
       Sleep for at least the given duration. Note that the underlying
       system will round the interval up to an integer multiple of the
       system's sleep resolution.
     */
    void sleep(Duration_t);

    /**
       Sleep until at least the given TimePoint_t.
    */
    void sleepUntil(TimePoint_t);

    /**
       Converts a TimePoint_t into a string.
       Note: the string formatting is used by the file summary catalog and
       may or may not depend on the actual formatting
    */
    std::string timeStamp(TimePoint_t);

    /**
       Converts a TimePoint_t into a string containg only the time in UTC.
    */
    std::string timeStampUTC(TimePoint_t);

    /**
       Converts a TimePoint_t into a string containing the time in UTC
       formatted as "Www Mmm dd hh:mm:ss yyyy"
    */
    std::string asctimeUTC(TimePoint_t);

    /**
       Converts a TimePoint_t into a string containing only the date.
       Note: the string formatting is used for file db log file name
    */
    std::string dateStamp(TimePoint_t);

    /**
       Returns an identifier string composed of class name and instance
    */
    std::string getIdentifier(xdaq::ApplicationDescriptor*);
    
    /**
       Throws a stor::exception::NoSuchDirectory when the directory does not exist
     */
    void checkDirectory(const std::string&);
 
    /**
       Conversions between std::vector<std::string> and xdata::Vector<xdata::String>
    */
    void getStdVector(xdata::Vector<xdata::String>&, std::vector<std::string>&);
    void getXdataVector(const std::vector<std::string>&, xdata::Vector<xdata::String>&);

    /**
       Compare items pointed to instead of pointer addresses
    */
    template<typename T, class Comp = std::less<T> > 
    struct ptrComp 
    { 
      bool operator()( T const* const lhs, T const* const rhs ) const 
      { 
        return comp( *lhs, *rhs ); 
      } 
      bool operator()( boost::shared_ptr<T> const& lhs, boost::shared_ptr<T> const& rhs ) const 
      { 
        return comp( *lhs, *rhs ); 
      } 
    private: 
      Comp comp; 
    }; 


    ///////////////////////////////////////
    // inline most commonly used methods //
    ///////////////////////////////////////

    inline Duration_t secondsToDuration(double const& seconds)
    {
      const unsigned int fullSeconds = static_cast<unsigned int>(seconds);
      return boost::posix_time::seconds(fullSeconds)
        + boost::posix_time::millisec(static_cast<unsigned int>((seconds - fullSeconds)*1000) );
    }
    
    inline double durationToSeconds(Duration_t const& duration)
    {
      return static_cast<double>(duration.total_milliseconds()) / 1000;
    }

    inline long secondsSinceEpoch(TimePoint_t const& timestamp)
    {
      const static boost::posix_time::ptime epoch(boost::gregorian::date(1970,1,1));
      return (timestamp - epoch).total_seconds();
    }

    inline TimePoint_t getCurrentTime()
    {
      return boost::posix_time::ptime(boost::posix_time::microsec_clock::universal_time());
    }

    inline void sleep(Duration_t interval)
    {
      boost::this_thread::sleep(interval); 
    }

    inline void sleepUntil(TimePoint_t theTime)
    {
      boost::this_thread::sleep(theTime);
    }

  } // namespace utils
  
} // namespace stor

#endif // EventFilter_StorageManager_Utils_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
