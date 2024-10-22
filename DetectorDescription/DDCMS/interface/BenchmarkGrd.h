#ifndef DETECTOR_DESCRIPTION_BENCHMARK_GRD_H
#define DETECTOR_DESCRIPTION_BENCHMARK_GRD_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <chrono>

class BenchmarkGrd {
public:
  BenchmarkGrd(const std::string &name) : m_start(std::chrono::high_resolution_clock::now()), m_name(name) {}

  ~BenchmarkGrd() {
    std::chrono::duration<double, std::milli> diff = std::chrono::high_resolution_clock::now() - m_start;
    edm::LogVerbatim("Geometry") << "Benchmark '" << m_name << "' took " << diff.count() << " millis\n";
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
  std::string m_name;
};

#ifdef BENCHMARK_ON
#define BENCHMARK_START(X) \
  {                        \
    BenchmarkGrd(#X)
#define BENCHMARK_END }
#else
#define BENCHMARK_START(X)
#define BENCHMARK_END
#endif

#endif
