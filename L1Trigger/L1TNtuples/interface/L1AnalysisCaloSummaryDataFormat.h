#ifndef L1Trigger_L1TNtuples_L1AnalysisCaloSummaryDataFormat_h
#define L1Trigger_L1TNtuples_L1AnalysisCaloSummaryDataFormat_h

namespace L1Analysis {
  struct L1AnalysisCaloSummaryDataFormat {
    L1AnalysisCaloSummaryDataFormat() { Reset(); }
    ~L1AnalysisCaloSummaryDataFormat(){};

    void Reset() {
      CICADAScore = 0;
      for (short iPhi = 0; iPhi < 18; ++iPhi)
        for (short iEta = 0; iEta < 14; ++iEta)
          modelInput[iPhi][iEta] = 0;
    }
    void Init() {}

    float CICADAScore;
    unsigned short int modelInput[18][14];  //Stored in indices of [iPhi][iEta]
  };
}  // namespace L1Analysis

#endif