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

    unsigned nMuons;
    std::vector<double> e;
    std::vector<double> et;
    std::vector<double> pt;
    std::vector<double> eta;
    std::vector<double> phi;
    std::vector<bool> isLooseMuon;
    std::vector<bool> isMediumMuon;
    std::vector<double> iso;
    std::vector<int> hlt_isomu;
    std::vector<int> hlt_mu;
    std::vector<double> hlt_isoDeltaR;
    std::vector<double> hlt_deltaR;

  };
}
#endif


