#ifndef __L1Analysis_L1AnalysisRecoMuon2DataFormat_H__
#define __L1Analysis_L1AnalysisRecoMuon2DataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
//
//
// Original code : L1Trigger/L1TNtuples/L1RecoMuon2NtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoMuon2DataFormat
  {
    L1AnalysisRecoMuon2DataFormat(){Reset();};
    ~L1AnalysisRecoMuon2DataFormat(){Reset();};

    void Reset()
    {
    nMuons=0;

    e.clear();
    et.clear();
    pt.clear();
    eta.clear();
    phi.clear();
    isLooseMuon.clear();
    isMediumMuon.clear();
    iso.clear();
    hlt_isomu.clear();
    hlt_mu.clear();
    hlt_isoDeltaR.clear();
    hlt_deltaR.clear();
    }

    unsigned short nMuons;
    std::vector<float> e;
    std::vector<float> et;
    std::vector<float> pt;
    std::vector<float> eta;
    std::vector<float> phi;
    std::vector<bool> isLooseMuon;
    std::vector<bool> isMediumMuon;
    std::vector<float> iso;
    std::vector<short> hlt_isomu;
    std::vector<short> hlt_mu;
    std::vector<float> hlt_isoDeltaR;
    std::vector<float> hlt_deltaR;

  };
}
#endif


