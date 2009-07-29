#ifndef CondCommon_TimeConversions_h
#define CondCommon_TimeConversions_h 

#include "CondFormats/Common/interface/Time.h"
#include <ctime>
#include <sys/time.h>
#include <string>
// FIXME incompatible with coral
// #define BOOST_DATE_TIME_POSIX_TIME_STD_CONFIG
#include "boost/date_time/posix_time/posix_time.hpp"

namespace cond {


  namespace time {

    
    typedef boost::date_time::subsecond_duration<boost::posix_time::time_duration,1000000000> nanoseconds;


    // valid for all representations

    const Time_t kLowMask(0xFFFFFFFF);
 
    inline cond::UnpackedTime unpack(cond::Time_t iValue) {
      return cond::UnpackedTime(iValue >> 32, kLowMask & iValue);
    }

    inline cond::Time_t pack(cond::UnpackedTime iValue) {
      Time_t t = iValue.first;
      return (t<< 32) + iValue.second;
    }


    // for real time 

    int itsNanoseconds(boost::posix_time::time_duration const & td) {
      boost::posix_time::time_duration::num_fractional_digits() == 6 ? 
	1000*td.fractional_seconds() : td.fractional_seconds() );
    }

    const boost::posix_time::ptime time0 =
      boost::posix_time::from_time_t(0);
    
    inline boost::posix_time::ptime to_boost(Time_t iValue) {
      return time0 +  
	boost::posix_time::seconds( iValue >> 32) + 
	nanoseconds(kLowMask & iValue);
    }
    
    
    inline Time_t from_boost(boost::posix_time::ptime bt) {
      boost::posix_time::time_duration td = bt - time0;
      Time_t t = td.total_seconds();
      return (t << 32) + itsNanoseconds(td);

    } 

    
    inline ::timeval to_timeval(Time_t iValue) {
      ::timeval stv;
      //  stv.tv_sec =  static_cast<unsigned int>(iValue >> 32);
      //stv.tv_usec = static_cast<unsigned int>(kLowMask & iValue);
      stv.tv_sec =  iValue >> 32;
      stv.tv_usec = (kLowMask & iValue)/1000;
      return stv;
    }
    
    
    inline Time_t from_timeval( ::timeval stv) {
      Time_t t = stv.tv_sec;
      return (t << 32) + 1000*stv.tv_usec;
    }
    
    inline Time_t now() {
      ::timeval stv;
      ::gettimeofday(&stv,0);
      return  from_timeval(stv);
    }
    
    
  } // ns time

} // cond


#endif
