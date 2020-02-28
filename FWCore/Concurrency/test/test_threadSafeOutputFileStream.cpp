#include "FWCore/Concurrency/interface/ThreadSafeOutputFileStream.h"

#include <atomic>
#include <sstream>
#include <thread>
#include <vector>

namespace {

  std::atomic<int> nWaitingForThreads{};

  void logToFile(unsigned const thread_index, edm::ThreadSafeOutputFileStream& f) {
    --nWaitingForThreads;
    while (nWaitingForThreads != 0)
      ;  // Spin until all threads have been launched.

    std::ostringstream oss;
    oss << "Thread index: " << thread_index << " Entry: ";
    auto const& prefix = oss.str();
    for (int i{}; i < 4; ++i) {
      f.write(prefix + std::to_string(i) + "\n");
    }
  }
}  // namespace

int main() {
  edm::ThreadSafeOutputFileStream f{"thread_safe_ofstream_test.txt"};
  std::vector<std::thread> threads;
  unsigned const nThreads{3};
  nWaitingForThreads = nThreads;
  for (unsigned i{}; i < nThreads; ++i) {
    threads.emplace_back(logToFile, i, std::ref(f));
  }

  for (auto& thread : threads)
    thread.join();
}
