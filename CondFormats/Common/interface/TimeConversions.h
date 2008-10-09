#ifndef CondCommon_TimeConversions_h
#define CondCommon_TimeConversions_h 

#include "CondFormats/Common/interface/Time.h"
#include <ctime>
#include <sys/time.h>
#include <string>
#define BOOST_DATE_TIME_POSIX_TIME_STD_CONFIG
#include "boost/date_time/posix_time/posix_time.hpp"

namespace cond {


  namespace time {


    // valid for all representations

    const Time_t kLowMask(0xFFFFFFFF);
 
    inline cond::UnpackedTime unpack(cond::Time_t iValue) {
      return UnpackedTime(iValue >> 32, kLowMask & iValue);
    }

    inline cond::Time_t pack(cond::UnpackedTime iValue) {
      return (iValue.first<< 32) + iValue.second;
    }


    // for real time 

    const boost::posix_time::ptime time0 =
      return boost::posix_time::from_time_t(0);
    
    inline boost::posix_time::ptime to_boost(Time_t iValue) {
      return time0 +  
	boost::posix_time::seconds( iValue >> 32) + 
	boost::posix_time::nanoseconds(kLowMask & iValue);
    }
    
    
    inline Time_t from_boost(boost::posix_time::ptime bt) {
      boost::posix_time::time_duration td = bt - time0;
      return (td.total_seconds() << 32) + td.fractional_seconds();

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
      return (stv.tv_sec << 32) + 1000*stv.tv_usec;
    }
    
    inline Time_t now() {
      ::timeval stv;
      ::gettimeofday(&stv,0);
      return  from_timeval(stv);
    }
    
    
  } // ns time

} // cond


#endif
