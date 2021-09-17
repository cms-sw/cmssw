#include "FWCore/SharedMemory/interface/WorkerMonitorThread.h"
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
  edm::shared_memory::WorkerMonitorThread monitor;

  monitor.startThread();

  monitor.setAction([&]() { std::cerr << "Action run\n"; });
  if (argc > 1) {
    char* end;
    int sig = std::strtol(argv[1], &end, 10);
    raise(sig);
  }
  return 0;
}
