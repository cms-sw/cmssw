#ifndef __L1Analysis_L1AnalysisRecoTauDataFormat_H__
#define __L1Analysis_L1AnalysisRecoTauDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
//
//
// Original code : L1Trigger/L1TNtuples/L1RecoTauNtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoTauDataFormat
  {
    L1AnalysisRecoTauDataFormat(){Reset();};
    ~L1AnalysisRecoTauDataFormat(){Reset();};

    void Reset()
    {
    nTaus=0;

    e.clear();
    et.clear();
    pt.clear();
    eta.clear();
    phi.clear();
    TightIsoFlag.clear();
    LooseIsoFlag.clear();
    LooseAntiMuonFlag.clear();
    TightAntiMuonFlag.clear();
    VLooseAntiElectronFlag.clear();
    LooseAntiElectronFlag.clear();
    TightAntiElectronFlag.clear();
    DMFindingNewDMs.clear();
    DMFindingOldDMs.clear();
    }

    unsigned nTaus;
    std::vector<float> e;
    std::vector<float> et;
    std::vector<float> pt;
    std::vector<float> eta;
    std::vector<float> phi;
    std::vector<float> DMFindingNewDMs;
    std::vector<float> DMFindingOldDMs;
    std::vector<float> TightIsoFlag;
    std::vector<float> LooseIsoFlag;
    std::vector<float> LooseAntiMuonFlag;
    std::vector<float> TightAntiMuonFlag;
    std::vector<float> VLooseAntiElectronFlag;
    std::vector<float> LooseAntiElectronFlag;
    std::vector<float> TightAntiElectronFlag;

  };
}
#endif


