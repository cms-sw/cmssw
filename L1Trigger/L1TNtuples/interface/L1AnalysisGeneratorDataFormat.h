#ifndef __L1Analysis_L1AnalysisGeneratorDataFormat_H__
#define __L1Analysis_L1AnalysisGeneratorDataFormat_H__

//-------------------------------------------------------------------------------
// Created 15/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------
#include <TROOT.h>
#include <vector>
//#include <TString.h>


namespace L1Analysis
{
  struct L1AnalysisGeneratorDataFormat
  {
  
    L1AnalysisGeneratorDataFormat(){Reset();};
    ~L1AnalysisGeneratorDataFormat(){};
    
    void Reset()
    {
     weight = -999.;
     pthat  = -999.;
     nVtyx  = 0;

     partId.resize(0);
     partStat.resize(0);
     partParent.resize(0);
     partPt.resize(0);
     partEta.resize(0);
     partPhi.resize(0);
     partE.resize(0);
     
     jetPt.resize(0);
     jetEta.resize(0);
     jetPhi.resize(0);
     jetM.resize(0);

    }

                   
    // ---- L1AnalysisGeneratorDataFormat information.
    
    float weight;
    float pthat;
    int nVTx;

    std::vector<int> partId;
    std::vector<int> poartStat;
    std::vector<int> partParent;
    std::vector<float> partPt;
    std::vector<float> partEta;
    std::vector<float> partPhi;
    std::vector<float> partM;
    
    std::vector<float> jetPt;
    std::vector<float> jetEta;
    std::vector<float> jetPhi;
    std::vector<float> jetM;

  }; 
} 
#endif


