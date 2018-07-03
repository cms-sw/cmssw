#ifndef __L1Analysis_L1AnalysisGTDataFormat_H__
#define __L1Analysis_L1AnalysisGTDataFormat_H__

//-------------------------------------------------------------------------------
// Created 15/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------
#include <vector>
// #include <inttypes.h>
#include <TROOT.h>

namespace L1Analysis
{
  struct L1AnalysisGTDataFormat
  {
    L1AnalysisGTDataFormat(){Reset();};
    ~L1AnalysisGTDataFormat(){};
    
    void Reset()
    { 
       tw1.clear();
       tw2.clear();
       tt.clear();

       partrig_tcs=0;
       gpsTimehi=0;
       gpsTimelo=0;
       bstMasterStatus=0; 
       bstturnCountNumber=0;
       bstlhcFillNumber=0;
       bstbeamMode=0;	
       bstparticleTypeBeam1=0;   
       bstparticleTypeBeam2=0;   
       bstbeamMomentum=0;   
       bsttotalIntensityBeam1=0;   
       bsttotalIntensityBeam2=0; 

    	//PSB info
       Nele = 0;
       Bxel.clear();
       Rankel.clear();
       Phiel.clear();
       Etael.clear();
       Isoel.clear();
       
       Njet = 0;
       Bxjet.clear();
       Rankjet.clear();
       Phijet.clear();
       Etajet.clear();
       Taujet.clear();
       Fwdjet.clear();  

// ------ ETT, ETM, HTT and HTM from PSB14:

	RankETT = -1;
	OvETT = false;
	RankHTT = -1;
	OvHTT = false;
	RankETM = -1;
	PhiETM = -1;
	OvETM = false;
	RankHTM = -1;
	PhiHTM = -1;
	OvHTM = false;
    }
   
    // ---- L1AnalysisGTDataFormat information.
    
    std::vector<ULong64_t> tw1;
    std::vector<ULong64_t> tw2;
    std::vector<ULong64_t> tt;
    unsigned long partrig_tcs;
    unsigned long gpsTimehi;
    unsigned long gpsTimelo;
    unsigned long bstMasterStatus; 
    unsigned long bstturnCountNumber;
    unsigned long bstlhcFillNumber;
    unsigned long bstbeamMode;   
    unsigned long bstparticleTypeBeam1;   
    unsigned long bstparticleTypeBeam2;   
    unsigned long bstbeamMomentum;	
    unsigned long bsttotalIntensityBeam1;   
    unsigned long bsttotalIntensityBeam2; 
    
    //PSB info
    int            Nele;
    std::vector<int>    Bxel;
    std::vector<float>  Rankel;
    std::vector<float>  Phiel;
    std::vector<float>  Etael;
    std::vector<bool>   Isoel;
    
    int            Njet;
    std::vector<int>    Bxjet;
    std::vector<float>  Rankjet;
    std::vector<float>  Phijet;
    std::vector<float>  Etajet;
    std::vector<bool>   Taujet;
    std::vector<bool>   Fwdjet;

// ------ ETT, ETM, HTT and HTM from PSB14:

    int RankETT;
    bool OvETT;

    int RankHTT;
    bool OvHTT;

    int RankETM;
    int PhiETM;
    bool OvETM;

    int RankHTM;
    int PhiHTM;
    bool OvHTM;

  }; 
} 
#endif


