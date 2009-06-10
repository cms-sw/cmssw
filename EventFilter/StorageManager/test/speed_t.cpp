#include <iostream>
#include <cstdlib>
#include "boost/thread/mutex.hpp"
#include "FWCore/Utilities/interface/CPUTimer.h"

using namespace std;

boost::mutex global_mutex;

inline void do_it()
{
  boost::mutex::scoped_lock lock(global_mutex);
}

int main(int argc, char* argv[])
{
  edm::CPUTimer t;
  int nloops = 1000;
  if (argc > 1) nloops = atoi(argv[1]);
  cout << "Doing " << nloops << " loops" << endl;

  t.start();
  for (int i = 0; i != nloops; ++i)
    {
      do_it();
    }
  t.stop();

  cout << "Time per lock: " << t.cpuTime()/nloops << '\n';
  
}
