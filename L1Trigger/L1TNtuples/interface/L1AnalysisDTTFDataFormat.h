#ifndef __L1Analysis_L1AnalysisDTTFDataFormat_H__
#define __L1Analysis_L1AnalysisDTTFDataFormat_H__

//-------------------------------------------------------------------------------
// Created 16/04/2010 - E. Conte, A.C. Le Bihan
// 
//  
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include <vector>
#include "TMatrixD.h"


namespace L1Analysis
{
  struct L1AnalysisDTTFDataFormat
  {

    L1AnalysisDTTFDataFormat(){Reset();};
    ~L1AnalysisDTTFDataFormat(){};
    
    void Reset()
    {

    phSize = 0;  
     
    phBx.clear(); 
    phWh.clear(); 
    phSe.clear(); 
    phSt.clear(); 
    phAng.clear();
    phGlobPhi.clear(); ///
    phBandAng.clear();
    phCode.clear(); 
    phX.clear();
    phY.clear();
        
    
    thSize = 0;
     
    thBx.clear();
    thWh.clear();
    thSe.clear();
    thSt.clear();
    thX.clear(); 
    thY.clear(); 
    
    
    trSize = 0;
    
    trBx.clear(); 
    trTag.clear();
    trQual.clear(); 
    trPtPck.clear();
    trPtVal.clear();
    trPhiPck.clear(); 
    trPhiVal.clear();
    trEtaPck.clear(); 
    trEtaVal.clear();  
    trPhiGlob.clear(); 
    trChPck.clear();
    trWh.clear(); 
    trSc.clear(); 
    trAddress.clear();

    thTheta.Clear();
    thCode.Clear();
    }
      
    // ---- L1AnalysisDTTFDataFormat information.
    
    int phSize;    
    std::vector<int>    phBx; 
    std::vector<int>    phWh; 
    std::vector<int>    phSe; 
    std::vector<int>    phSt; 
    std::vector<float>  phAng;
    std::vector<double> phGlobPhi;
    std::vector<float>  phBandAng;
    std::vector<int>    phCode; 
    std::vector<float>  phX;
    std::vector<float>  phY;
    
    int thSize;
    std::vector<int>   thBx;
    std::vector<int>   thWh;
    std::vector<int>   thSe;
    std::vector<int>   thSt;
    std::vector<float> thX; 
    std::vector<float> thY; 
    
    TMatrixD thTheta;
    TMatrixD thCode; 
    
    int trSize;
    std::vector<int>   trBx; 
    std::vector<int>   trTag;
    std::vector<int>   trQual; 
    std::vector<int>   trPtPck;
    std::vector<float> trPtVal;
    std::vector<int>   trPhiPck; 
    std::vector<float> trPhiVal; 
    std::vector<int>   trEtaPck; 
    std::vector<float> trEtaVal; 
    std::vector<double>trPhiGlob; 
    std::vector<int>   trChPck;
    std::vector<int>   trWh; 
    std::vector<int>   trSc;  
    std::vector<unsigned int> trAddress;
  }; 
} 
#endif


