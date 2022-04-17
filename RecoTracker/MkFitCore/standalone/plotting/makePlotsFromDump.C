#include "plotting/PlotsFromDump.cpp+"

void makePlotsFromDump(const TString& sample, const TString& build, const TString& suite, const int useARCH) {
  PlotsFromDump Plots(sample, build, suite, useARCH);
  Plots.RunPlotsFromDump();
}
