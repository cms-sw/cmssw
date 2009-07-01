// $Id: OutputService.cc,v 1.5 2008/05/13 18:06:46 loizides Exp $

#include <EventFilter/StorageManager/interface/OutputService.h>
#include <iostream>
#include <sys/time.h> 
 
using namespace edm;

OutputService::~OutputService() {
  //std::cout << "OutputService Destructor called." << std::endl;
}


// 
// *** get the current time stamp
//
double OutputService::getTimeStamp() const
{
  struct timeval now;
  struct timezone dummyTZ;
  gettimeofday(&now, &dummyTZ);
  return (double) now.tv_sec + (double) now.tv_usec / 1000000.0;
}
