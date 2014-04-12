#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/IOVInfo.h"


#include <iostream>
#include <sys/time.h>
#include "DataFormats/Provenance/interface/Timestamp.h"
int main() {
  std::cout << cond::userInfo() << std::endl;
  ::timeval tv;
  gettimeofday(&tv,0);
  std::cout<<"sec "<<tv.tv_sec<<std::endl;
  std::cout<<"micro sec "<<tv.tv_usec<<std::endl;
  edm::Timestamp tmstamp((unsigned long long)tv.tv_sec*1000000+(unsigned long long)tv.tv_usec);
  std::cout<<"timestamp of the day since 1970 in microsecond "<<tmstamp.value()<<std::endl;
  edm::Timestamp tstamp((unsigned long long)tv.tv_sec*1000000);
  std::cout<<"timestamp of the day since 1970 in second "<<tstamp.value()/1000000<<std::endl;
  //from  IORawData/DaqSource/plugins/DaqSource.cc
  edm::TimeValue_t daqtime=0LL;
  ::timeval stv;
  gettimeofday(&stv,0);
  daqtime=stv.tv_sec;
  daqtime=(daqtime<<32)+stv.tv_usec;
  edm::Timestamp daqstamp(daqtime);
  std::cout<<"timestamp of the day since 1970 in DAQ "<<daqstamp.value()<<std::endl;
  //  edm::TimeValue_t bizzartime=4294967295LL;
  //usec=bizzartime
  using namespace cond;
  for (size_t i=0; i<TIMETYPE_LIST_MAX; i++) 
    std::cout << "Time Specs:" 
	      << " enum " << timeTypeSpecs[i].type
	      << ", name " << timeTypeSpecs[i].name
	      << ", begin " << timeTypeSpecs[i].beginValue
	      << ", end " << timeTypeSpecs[i].endValue
	      << ", invalid " << timeTypeSpecs[i].invalidValue
	      << std::endl;

  try {
    for (size_t i=0; i<TIMETYPE_LIST_MAX; i++)
      if (cond::findSpecs(timeTypeSpecs[i].name).type!=timeTypeSpecs[i].type)
	std::cout << "error in find for " << timeTypeSpecs[i].name << std::endl;
    
    cond::findSpecs("fake");
  }
  catch(cms::Exception const & e) {
    std::cout << "Expected error: "<< e.what() << std::endl;
  }

  return 0;
}
