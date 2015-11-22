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
    }

    unsigned nElectrons;
    std::vector<double> e;
    std::vector<double> et;
    std::vector<double> e_SC;
    std::vector<double> e_ECAL;
    std::vector<double> phi_SC;
    std::vector<double> pt;
    std::vector<double> eta;
    std::vector<double> eta_SC;
    std::vector<double> phi;
    std::vector<double> iso;
    std::vector<int> isVetoElectron;
    std::vector<int> isLooseElectron;
    std::vector<int> isMediumElectron;
    std::vector<int> isTightElectron;

  };
}
#endif


