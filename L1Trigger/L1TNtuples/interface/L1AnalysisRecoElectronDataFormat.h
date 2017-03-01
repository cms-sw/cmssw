#ifndef __L1Analysis_L1AnalysisRecoElectronDataFormat_H__
#define __L1Analysis_L1AnalysisRecoElectronDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
//
//
// Original code : L1Trigger/L1TNtuples/L1RecoElectronNtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoElectronDataFormat
  {
    L1AnalysisRecoElectronDataFormat(){Reset();};
    ~L1AnalysisRecoElectronDataFormat(){Reset();};

    void Reset()
    {
    nElectrons=0;

    e.clear();
    et.clear();
    e_ECAL.clear();
    e_SC.clear();
    pt.clear();
    eta.clear();
    eta_SC.clear();
    phi_SC.clear();
    phi.clear();
    iso.clear();
    isVetoElectron.clear();
    isLooseElectron.clear();
    isMediumElectron.clear();
    isTightElectron.clear();
    charge.clear();
    }

    unsigned nElectrons;
    std::vector<float> e;
    std::vector<float> et;
    std::vector<float> e_SC;
    std::vector<float> e_ECAL;
    std::vector<float> phi_SC;
    std::vector<float> pt;
    std::vector<float> eta;
    std::vector<float> eta_SC;
    std::vector<float> phi;
    std::vector<float> iso;
    std::vector<short> isVetoElectron;
    std::vector<short> isLooseElectron;
    std::vector<short> isMediumElectron;
    std::vector<short> isTightElectron;
    std::vector<int> charge;

  };
}
#endif


