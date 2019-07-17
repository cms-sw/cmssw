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
      nECALEBTP = 0;
      ecalEBTPieta.clear();
      ecalEBTPiphi.clear();
      ecalEBTPCaliphi.clear();
      ecalEBTPet.clear();
      ecalEBTPencodEt.clear();
      ecalEBTPspike.clear();
      ecalEBTPtime.clear();
    }
    
    void Init() {

    }
    

    short nHCALTP;
    std::vector<short> hcalTPieta;
    std::vector<short> hcalTPiphi;
    std::vector<short> hcalTPCaliphi;
    std::vector<float> hcalTPet;
    std::vector<short> hcalTPcompEt;
    std::vector<short> hcalTPfineGrain;

    short nECALTP;
    std::vector<short> ecalTPieta;
    std::vector<short> ecalTPiphi;
    std::vector<short> ecalTPCaliphi;
    std::vector<float> ecalTPet;
    std::vector<short> ecalTPcompEt;
    std::vector<short> ecalTPfineGrain;

    short nECALEBTP;
    std::vector<short> ecalEBTPieta;
    std::vector<short> ecalEBTPiphi;
    std::vector<short> ecalEBTPCaliphi;
    std::vector<float> ecalEBTPet;
    std::vector<short> ecalEBTPencodEt;
    std::vector<short> ecalEBTPspike;
    std::vector<short> ecalEBTPtime;
    
    
  }; 
} 
#endif

