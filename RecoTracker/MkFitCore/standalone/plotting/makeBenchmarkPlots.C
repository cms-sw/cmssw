#include "plotting/PlotBenchmarks.cpp+"

void makeBenchmarkPlots(const TString& arch, const TString& sample, const TString& suite) {
  PlotBenchmarks Benchmarks(arch, sample, suite);
  Benchmarks.RunBenchmarkPlots();
}
