#include "FWCore/Concurrency/interface/ThreadSafeOutputFileStream.h"

#include <sstream>
#include <thread>
#include <vector>

namespace {
  void
  logToFile(unsigned const thread_index, edm::ThreadSafeOutputFileStream& f)
  {
    std::ostringstream oss;
    oss << "Thread index: " << thread_index << " Entry: ";
    auto const& prefix = oss.str();
    for (int i{}; i < 4; ++i) {
      std::string const msg {prefix+std::to_string(i)};
      f << msg;
    }
  }
}

int
main()
{
  edm::ThreadSafeOutputFileStream f {"thread_safe_ofstream_test.txt"};
  std::vector<std::thread> threads;
  threads.emplace_back(logToFile, 0, std::ref(f));
  threads.emplace_back(logToFile, 1, std::ref(f));
  threads.emplace_back(logToFile, 2, std::ref(f));

  for (auto& thread : threads)
    thread.join();
}
