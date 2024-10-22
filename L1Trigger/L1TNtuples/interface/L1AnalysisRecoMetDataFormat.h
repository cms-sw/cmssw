#ifndef __L1Analysis_L1AnalysisRecoMetDataFormat_H__
#define __L1Analysis_L1AnalysisRecoMetDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
//
//
// Addition of met reco information
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis {
  struct L1AnalysisRecoMetDataFormat {
    L1AnalysisRecoMetDataFormat() { Reset(); };
    ~L1AnalysisRecoMetDataFormat() { Reset(); };

    void Reset() {
      met = -999.;
      metPx = -999.;
      metPy = -999.;
      metPhi = -999.;
      pfMetNoMu = -999.;
      pfMetNoMuPx = -999.;
      pfMetNoMuPy = -999.;
      pfMetNoMuPhi = -999.;
      puppi_metNoMu = -999.;
      puppi_metNoMuPx = -999.;
      puppi_metNoMuPy = -999.;
      puppi_metNoMuPhi = -999.;
      caloMet = -999.;
      caloMetPhi = -999.;
      caloSumEt = -999.;
      caloMetBE = -999.;
      caloMetPhiBE = -999.;
      caloSumEtBE = -999.;
      caloHt = -999.;
      Ht = -999.;
      mHt = -999.;
      mHtPhi = -999.;
      puppi_Ht = -999.;
      puppi_mHt = -999.;
      puppi_mHtPhi = -999.;
      sumEt = -999.;
      zPt = -999.;
      ecalFlag = 0;
      hcalFlag = 0;
    }

    float met;
    float metPx;
    float metPy;
    float metPhi;
    float puppi_metNoMu;
    float puppi_metNoMuPx;
    float puppi_metNoMuPy;
    float puppi_metNoMuPhi;
    float pfMetNoMu;
    float pfMetNoMuPx;
    float pfMetNoMuPy;
    float pfMetNoMuPhi;
    float caloMet;
    float caloMetPhi;
    float caloSumEt;
    float caloMetBE;
    float caloMetPhiBE;
    float caloSumEtBE;
    float caloHt;
    float Ht;
    float mHt;
    float mHtPhi;
    float sumEt;
    float puppi_Ht;
    float puppi_mHt;
    float puppi_mHtPhi;
    float zPt;
    unsigned short ecalFlag;
    unsigned short hcalFlag;
  };
}  // namespace L1Analysis
#endif
