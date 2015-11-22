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

    /* e.clear(); */
    /* et.clear(); */
    /* etCorr.clear(); */
    /* corrFactor.clear(); */
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
    /* phi.clear(); */
    /* eEMF.clear(); */
    /* eEmEB.clear(); */
    /* eEmEE.clear(); */
    /* eEmHF.clear(); */
    /* eHadHB.clear(); */
    /* eHadHE.clear(); */
    /* eHadHO.clear(); */
    /* eHadHF.clear(); */
    /* eMaxEcalTow.clear(); */
    /* eMaxHcalTow.clear(); */
    /* towerArea.clear(); */
    /* towerSize.clear(); */
    /* n60.clear(); */
    /* n90.clear(); */

    /* n90hits.clear(); */
    /* fHPD.clear(); */
    /* fRBX.clear(); */
    }

    unsigned nTaus;
    /* std::vector<double> e; */
    /* std::vector<double> et; */
    /* std::vector<double> etCorr; */
    /* std::vector<double> corrFactor; */
    std::vector<double> e;
    std::vector<double> et;
    std::vector<double> pt;
    std::vector<double> eta;
    std::vector<double> phi;
    std::vector<double> DMFindingNewDMs;
    std::vector<double> DMFindingOldDMs;
    std::vector<double> TightIsoFlag;
    std::vector<double> LooseIsoFlag;
    std::vector<double> LooseAntiMuonFlag;
    std::vector<double> TightAntiMuonFlag;
    std::vector<double> VLooseAntiElectronFlag;
    std::vector<double> LooseAntiElectronFlag;
    std::vector<double> TightAntiElectronFlag;
    /* std::vector<double> phi; */
    /* std::vector<double> eEMF; */
    /* std::vector<double> eHadHB; */
    /* std::vector<double> eHadHE; */
    /* std::vector<double> eHadHO; */
    /* std::vector<double> eHadHF; */
    /* std::vector<double> eEmEB; */
    /* std::vector<double> eEmEE; */
    /* std::vector<double> eEmHF; */
    /* std::vector<double> eMaxEcalTow; */
    /* std::vector<double> eMaxHcalTow; */
    /* std::vector<double> towerArea; */
    /* std::vector<int> towerSize; */
    /* std::vector<int> n60; */
    /* std::vector<int> n90; */

    /* std::vector<int> n90hits; */
    /* std::vector<double> fHPD; */
    /* std::vector<double> fRBX; */

  };
}
#endif


