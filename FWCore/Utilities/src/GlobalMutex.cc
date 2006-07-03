#include "FWCore/Utilities/interface/GlobalMutex.h"
#include "boost/thread/mutex.hpp"

boost::mutex* edm::rootfix::getGlobalMutex()
  { 
    static boost::mutex m_;
    return &m_;
  }
