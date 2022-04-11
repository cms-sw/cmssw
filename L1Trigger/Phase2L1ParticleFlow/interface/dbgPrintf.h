#ifndef L1Trigger_Phase2L1ParticleFlow_dbgPrintf_h
#define L1Trigger_Phase2L1ParticleFlow_dbgPrintf_h

#ifdef CMSSW_GIT_HASH
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

inline edm::LogPrint dbgCout() { return edm::LogPrint("L1TCorrelator"); }
inline edm::LogProblem dbgCerr() { return edm::LogProblem("L1TCorrelator"); }

template <typename... Args>
inline void dbgPrintf(const char *formatString, Args &&...args) {
  char buff[1024];
  std::fill(buff, buff + 1024, '\0');
  int ret = snprintf(buff, 1023, formatString, std::forward<Args>(args)...);
  if (ret > 0 && ret < 1023 && buff[ret - 1] == '\n')
    buff[ret - 1] = '\0';
  edm::LogPrint("L1TCorrelator") << std::string_view(buff);
}

#else  // outside CMSSW: just use std::cout and printf

#include <iostream>

inline std::ostream &dbgCout() { return std::cout; }
inline std::ostream &dbgCerr() { return std::cerr; }

template <typename... Args>
inline void dbgPrintf(const char *formatString, Args &&...args) {
  printf(formatString, std::forward<Args>(args)...);
}

#endif

#endif
