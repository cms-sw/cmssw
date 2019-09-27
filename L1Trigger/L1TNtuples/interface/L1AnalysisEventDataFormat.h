#ifndef __L1Analysis_L1AnalysisEventDataFormat_H__
#define __L1Analysis_L1AnalysisEventDataFormat_H__

//-------------------------------------------------------------------------------
// Created 15/04/2010 - E. Conte, A.C. Le Bihan
//
//
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

// #include <inttypes.h>
#include <vector>
#include <TString.h>

namespace L1Analysis {
  struct L1AnalysisEventDataFormat {
    L1AnalysisEventDataFormat() { Reset(); }
    ~L1AnalysisEventDataFormat() {}

    void Reset() {
      run = -1;
      event = -1;
      lumi = -1;
      bx = -1;
      orbit = 0;
      time = 0;
      hlt.resize(0);
    }

    unsigned run;
    unsigned long long event;
    unsigned lumi;
    unsigned bx;
    //uint64_t orbit;
    ULong64_t orbit;
    //uint64_t time;
    ULong64_t time;
    int nPV;
    int nPV_True;
    std::vector<TString> hlt;

    double puWeight;
  };
}  // namespace L1Analysis
#endif
