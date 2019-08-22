#ifndef __L1Analysis_L1AnalysisRecoMetFilterDataFormat_H__
#define __L1Analysis_L1AnalysisRecoMetFilterDataFormat_H__

//-------------------------------------------------------------------------------
// Created 27/01/2016 - A. Bundock
//
//
// Addition of met reco information
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis {
  struct L1AnalysisRecoMetFilterDataFormat {
    L1AnalysisRecoMetFilterDataFormat() { Reset(); };
    ~L1AnalysisRecoMetFilterDataFormat() { Reset(); };

    void Reset() {
      hbheNoiseFilter = false;
      hbheNoiseIsoFilter = false;
      cscTightHalo2015Filter = false;
      globalSuperTightHalo2016Filter = false;
      ecalDeadCellTPFilter = false;
      goodVerticesFilter = false;
      eeBadScFilter = false;
      chHadTrackResFilter = false;
      muonBadTrackFilter = false;
      badPFMuonFilter = false;
      badChCandFilter = false;
    }

    bool hbheNoiseFilter;
    bool hbheNoiseIsoFilter;
    bool cscTightHalo2015Filter;
    bool globalSuperTightHalo2016Filter;
    bool ecalDeadCellTPFilter;
    bool goodVerticesFilter;
    bool eeBadScFilter;
    bool chHadTrackResFilter;
    bool muonBadTrackFilter;
    bool badPFMuonFilter;
    bool badChCandFilter;
  };
}  // namespace L1Analysis
#endif
