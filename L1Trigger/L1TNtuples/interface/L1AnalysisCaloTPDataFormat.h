#ifndef __L1Analysis_L1AnalysisCaloTPDataFormat_H__
#define __L1Analysis_L1AnalysisCaloTPDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisCaloTPDataFormat
  {
    L1AnalysisCaloTPDataFormat(){Reset();};
    ~L1AnalysisCaloTPDataFormat(){};
    
    
    void Reset() {
      nHCALTP = 0;
      hcalTPieta.clear();
      hcalTPiphi.clear();
      hcalTPCaliphi.clear();
      hcalTPet.clear();
      hcalTPcompEt.clear();
      hcalTPfineGrain.clear();
      nECALTP = 0;
      ecalTPieta.clear();
      ecalTPiphi.clear();
      ecalTPCaliphi.clear();
      ecalTPet.clear();
      ecalTPcompEt.clear();
      ecalTPfineGrain.clear();
    }
    
    void Init() {

    }
    

    int nHCALTP;
    std::vector<int> hcalTPieta;
    std::vector<int> hcalTPiphi;
    std::vector<int> hcalTPCaliphi;
    std::vector<double> hcalTPet;
    std::vector<int> hcalTPcompEt;
    std::vector<short int> hcalTPfineGrain;

    int nECALTP;
    std::vector<int> ecalTPieta;
    std::vector<int> ecalTPiphi;
    std::vector<int> ecalTPCaliphi;
    std::vector<double> ecalTPet;
    std::vector<int> ecalTPcompEt;
    std::vector<short int> ecalTPfineGrain;
    
  }; 
} 
#endif

