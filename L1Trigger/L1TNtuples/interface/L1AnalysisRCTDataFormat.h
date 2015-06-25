#ifndef __L1Analysis_L1AnalysisRCTDataFormat_H__
#define __L1Analysis_L1AnalysisRCTDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRCTDataFormat
  {
    //  L1AnalysisRCTDataFormat(){Reset();};
    L1AnalysisRCTDataFormat(){Reset();};
    ~L1AnalysisRCTDataFormat(){};
    
    void Reset()
    {
    RegSize=-999;

    RegEta.clear();
    RegPhi.clear();
    RegGEta.clear();
    RegGPhi.clear();
    RegRnk.clear();
    RegVeto.clear();
    RegBx.clear();
    RegOverFlow.clear();
    RegMip.clear();
    RegFGrain.clear();

    EmSize=-999;

    IsIsoEm.clear();
    EmEta.clear();
    EmPhi.clear();
    EmRnk.clear();
    EmBx.clear();


    }

    void InitHdRCT()
    {
    RegSize=-999;

    RegEta.assign(maxRCTREG_,-999.);
    RegPhi.assign(maxRCTREG_,-999.);
    RegGEta.assign(maxRCTREG_,-999.);
    RegGPhi.assign(maxRCTREG_,-999.);
    RegRnk.assign(maxRCTREG_,-999.);
    RegVeto.assign(maxRCTREG_,-999);
    RegBx.assign(maxRCTREG_,-999);
    RegOverFlow.assign(maxRCTREG_,-999);
    RegMip.assign(maxRCTREG_,-999);
    RegFGrain.assign(maxRCTREG_,-999);
    }

    void InitEmRCT()
    {
    EmSize=-999;

    IsIsoEm.assign(maxRCTREG_,-999);
    EmEta.assign(maxRCTREG_,-999.);
    EmPhi.assign(maxRCTREG_,-999.);
    EmRnk.assign(maxRCTREG_,-999.);
    EmBx.assign(maxRCTREG_,-999-maxRCTREG_); 
    }
     
    // ---- L1AnalysisRCTDataFormat information.
    int maxRCTREG_;
    
    int RegSize;
    
    std::vector<float> RegEta;
    std::vector<float> RegPhi;
    std::vector<float> RegGEta;
    std::vector<float> RegGPhi;
    std::vector<float> RegRnk;
    std::vector<int>   RegVeto;
    std::vector<int>   RegBx;
    std::vector<int>   RegOverFlow;
    std::vector<int>   RegMip;
    std::vector<int>   RegFGrain;   
   
 
    int EmSize;
     
    std::vector<int>   IsIsoEm;
    std::vector<float> EmEta;
    std::vector<float> EmPhi;
    std::vector<float> EmRnk;
    std::vector<int>   EmBx;
   
   
  }; 
} 
#endif


