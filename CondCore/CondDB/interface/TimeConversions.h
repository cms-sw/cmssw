#ifndef CondCore_CondDB_TimeConversions_h
#define CondCore_CondDB_TimeConversions_h 
//
#include "CondCore/CondDB/interface/Time.h"
//
#include <ctime>
#include <sys/time.h>
#include <string>
//
#include "boost/date_time/posix_time/posix_time.hpp"

namespace conddb {

  namespace time {

    typedef boost::date_time::subsecond_duration<boost::posix_time::time_duration,1000000000> nanoseconds;


    // valid for all representations

    const conddb::Time_t kLowMask(0xFFFFFFFF);
 
    inline UnpackedTime unpack(conddb::Time_t iValue) {
      return UnpackedTime(iValue >> 32, kLowMask & iValue);
    }

    inline conddb::Time_t pack(conddb::UnpackedTime iValue) {
      conddb::Time_t t = iValue.first;
      return (t<< 32) + iValue.second;
    }


    // for real time 
    inline unsigned int itsNanoseconds(boost::posix_time::time_duration const & td) {
      return boost::posix_time::time_duration::num_fractional_digits() == 6 ? 
	1000*td.fractional_seconds() : td.fractional_seconds();
    }


    const boost::posix_time::ptime time0 =
      boost::posix_time::from_time_t(0);
    
    inline boost::posix_time::ptime to_boost(Time_t iValue) {
      return time0 +  
	boost::posix_time::seconds( iValue >> 32) + 
	nanoseconds(kLowMask & iValue);
    }
    
    
    inline conddb::Time_t from_boost(boost::posix_time::ptime bt) {
      boost::posix_time::time_duration td = bt - time0;
      conddb::Time_t t = td.total_seconds();
      return (t << 32) + itsNanoseconds(td);

    } 

    
    inline ::timeval to_timeval(conddb::Time_t iValue) {
      ::timeval stv;
      //  stv.tv_sec =  static_cast<unsigned int>(iValue >> 32);
      //stv.tv_usec = static_cast<unsigned int>(kLowMask & iValue);
      stv.tv_sec =  iValue >> 32;
      stv.tv_usec = (kLowMask & iValue)/1000;
      return stv;
    }
    
    
    inline conddb::Time_t from_timeval( ::timeval stv) {
      conddb::Time_t t = stv.tv_sec;
      return (t << 32) + 1000*stv.tv_usec;
    }
    
    inline conddb::Time_t now() {
      ::timeval stv;
      ::gettimeofday(&stv,0);
      return  from_timeval(stv);
    }
    
    conddb::Time_t getTill( conddb::Time_t nextSince, conddb::TimeType timeType ){
      if( timeType != time::TIMESTAMP ){
	return nextSince - 1;
      } else {
	time::UnpackedTime unpackedTime = conddb::time::unpack(  nextSince );
	//number of seconds in nanoseconds (avoid multiply and divide by 1e09)
	conddb::Time_t totalSecondsInNanoseconds = ((conddb::Time_t)unpackedTime.first)*1000000000;
	//total number of nanoseconds
	conddb::Time_t totalNanoseconds = totalSecondsInNanoseconds + ((conddb::Time_t)(unpackedTime.second));
	//now decrementing of 1 nanosecond
	totalNanoseconds--;
	//now repacking (just change the value of the previous pair)
	unpackedTime.first = (unsigned int) (totalNanoseconds/1000000000);
	unpackedTime.second = (unsigned int)(totalNanoseconds - (conddb::Time_t)unpackedTime.first*1000000000);
	return conddb::time::pack(unpackedTime);
      }
    }
    
  } // ns time

} // conddb


#endif
