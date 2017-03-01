#ifndef UCTLogging_hh
#define UCTLogging_hh

#define CMSSW

#ifdef CMSSW
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOG_ERROR edm::LogError("L1TCaloLayer1")
#else
#define LOG_ERROR std::cerr
#endif

#endif
