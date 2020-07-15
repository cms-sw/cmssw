// example of use of time conversions
#include "CondFormats/Common/interface/TimeConversions.h"

#include <iostream>

int main() {
  {
    std::cout << "boost frac digits " << boost::posix_time::time_duration::num_fractional_digits() << std::endl;
    ::timeval stv;
    ::gettimeofday(&stv, nullptr);

    cond::Time_t time = cond::time::from_timeval(stv);

    std::cout << stv.tv_sec << " " << stv.tv_usec << std::endl;
    std::cout << time << std::endl;
    stv = cond::time::to_timeval(time);
    std::cout << stv.tv_sec << " " << stv.tv_usec << std::endl;

    boost::posix_time::ptime bt = cond::time::to_boost(time);

    boost::posix_time::time_duration td = bt - cond::time::time0;

    std::cout << bt << std::endl;
    std::cout << "s. " << td.total_seconds() << "." << td.fractional_seconds() << std::endl;
    std::cout << "us " << td.total_microseconds() << std::endl;
    std::cout << "ns " << td.total_nanoseconds() << std::endl;
    std::cout << std::endl;

    // FIXME (when agree with Coral)
    //bt +=  boost::posix_time::nanoseconds(19*25);
    bt += cond::time::nanoseconds(19 * 25);

    td = bt - cond::time::time0;

    std::cout << bt << std::endl;
    std::cout << "s. " << td.total_seconds() << "." << td.fractional_seconds() << std::endl;
    std::cout << "us " << td.total_microseconds() << std::endl;
    std::cout << "ns " << td.total_nanoseconds() << std::endl;
    std::cout << std::endl;

    time = cond::time::from_boost(bt);
    cond::UnpackedTime utime = cond::time::unpack(time);
    std::cout << "s. " << utime.first << "." << utime.second << std::endl;
    stv = cond::time::to_timeval(time);
    std::cout << stv.tv_sec << " " << stv.tv_usec << std::endl;
  }

  return 0;
}
