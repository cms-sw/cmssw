// $Id: Utils.h,v 1.12 2010/12/14 12:56:51 mommsen Exp $
/// @file: Utils.h 

#ifndef StorageManager_Utils_h
#define StorageManager_Utils_h

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
     * $Revision: 1.12 $
     * $Date: 2010/12/14 12:56:51 $
     */

    /**
       time_point_t is used to represent a specific point in time
    */
    typedef boost::posix_time::ptime time_point_t;

    /**
       durtion_t is used to represent a duration (the "distance" between
       two points in time).
    */
    typedef boost::posix_time::time_duration duration_t;

    /**
       Convert a fractional second count into a boost::posix_time::time_duration
       type with a resolution of milliseconds.
    */
    duration_t seconds_to_duration(double const& seconds);

    /**
       Convert a boost::posix_time::time_duration into fractional seconds with
       a resolution of milliseconds.
    */
    double duration_to_seconds(duration_t const&);

    /**
       Return the number of seconds since the unix epoch 1-1-1970
    */
    long seconds_since_epoch(time_point_t const&);

    /**
       Returns the current point in time.
    */
    time_point_t getCurrentTime();

    /**
       Sleep for at least the given duration. Note that the underlying
       system will round the interval up to an integer multiple of the
       system's sleep resolution.
     */
    void sleep(duration_t);

    /**
       Sleep until at least the given time_point_t.
    */
    void sleepUntil(time_point_t);

    /**
       Converts a time_point_t into a string.
       Note: the string formatting is used by the file summary catalog and
       may or may not depend on the actual formatting
    */
    std::string timeStamp(time_point_t);

    /**
       Converts a time_point_t into a string containg only the time in UTC.
    */
    std::string timeStampUTC(time_point_t);

    /**
       Converts a time_point_t into a string containing the time in UTC
       formatted as "Www Mmm dd hh:mm:ss yyyy"
    */
    std::string asctimeUTC(time_point_t);

    /**
       Converts a time_point_t into a string containing only the date.
       Note: the string formatting is used for file db log file name
    */
    std::string dateStamp(time_point_t);

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
    struct ptr_comp 
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

    inline duration_t seconds_to_duration(double const& seconds)
    {
      const unsigned int fullSeconds = static_cast<unsigned int>(seconds);
      return boost::posix_time::seconds(fullSeconds)
        + boost::posix_time::millisec(static_cast<unsigned int>((seconds - fullSeconds)*1000) );
    }
    
    inline double duration_to_seconds(duration_t const& duration)
    {
      return static_cast<double>(duration.total_milliseconds()) / 1000;
    }

    inline long seconds_since_epoch(time_point_t const& timestamp)
    {
      const static boost::posix_time::ptime epoch(boost::gregorian::date(1970,1,1));
      return (timestamp - epoch).total_seconds();
    }
    
    inline time_point_t getCurrentTime()
    {
      return boost::posix_time::ptime(boost::posix_time::microsec_clock::universal_time());
    }

    inline void sleep(duration_t interval)
    {
      boost::this_thread::sleep(interval); 
    }

    inline void sleepUntil(time_point_t theTime)
    {
      boost::this_thread::sleep(theTime);
    }

  } // namespace utils
  
} // namespace stor

#endif // StorageManager_Utils_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
