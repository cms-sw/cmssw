#ifndef GlobalMutex_H
#define GlobalMutex_H 1

// Create a globally accessable mutex to be used for syncrhonization across 
// different packages that must use root in different threads.  This is to 
// be used with the LockService. Hopefully in the future this silliness
// can be removed.  -LSK
//class boost::mutex;
#include "boost/thread/mutex.hpp"
namespace edm {
  namespace rootfix {
    boost::mutex* getGlobalMutex();
  }
}
#endif
