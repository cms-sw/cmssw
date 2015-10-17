#ifndef __L1Analysis_L1AnalysisCSCTFDataFormat_H__
#define __L1Analysis_L1AnalysisCSCTFDataFormat_H__

//-------------------------------------------------------------------------------
// Created 08/03/2010 - E. Conte, A.-C. Le Bihan
//
//
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer -  Gian Piero Di Giovanni
//-------------------------------------------------------------------------------
#include "TMatrixD.h"
#include <vector>

namespace L1Analysis
{
  struct L1AnalysisCSCTFDataFormat
  {
    L1AnalysisCSCTFDataFormat(){Reset();};
    ~L1AnalysisCSCTFDataFormat(){};
    
    static const int MAXCSCTFTR = 60;
    static const int MAXCSCTFLCTSTR = 4;

    void Reset()
    {
    trSize = 0;
    trEndcap.clear(); 
    trSector.clear(); 
    
    trBx.clear(); 

    // if set to 1 track has lct from the station 
    trME1ID.clear(); 
    trME2ID.clear(); 
    trME3ID.clear(); 
    trME4ID.clear(); 
    trMB1ID.clear();     

    trME1TBin.clear();
    trME2TBin.clear();
    trME3TBin.clear();
    trME4TBin.clear();
    trMB1TBin.clear();

    trOutputLink.clear(); 
  
    // some input of the PT LUTs 
    trCharge.clear(); 
    trChargeValid.clear(); 
    trForR.clear(); 
    trPhi23.clear(); 
    trPhi12.clear();   
    trPhiSign.clear();   

    // in bits... 
    trEtaBit.clear();   
    trPhiBit.clear();   
    trPtBit.clear();   
  
    // ... converted 
    trEta.clear();   
    trPhi.clear();   
    trPhi_02PI.clear(); 
    trPt.clear();   

    // + useful information
    trMode.clear();
    trQuality.clear();

    //---------------------------------------------------------------------- 
    // LCT (STUBS FORMING THE TRACK)  
    //---------------------------------------------------------------------- 
    trNumLCTs.clear();
    
    trLctEndcap.Clear(); 
    trLctSector.Clear(); 
    trLctSubSector.Clear(); 
    trLctBx.Clear(); 
    trLctBx0.Clear(); 
   
    trLctStation.Clear(); 
    trLctRing.Clear(); 
    trLctChamber.Clear(); 
    trLctTriggerCSCID.Clear(); 
    trLctFpga.Clear();	 
    
    trLctlocalPhi.Clear(); 
    //trLctlocalPhi_bend.Clear(); 
    trLctCLCT_pattern.Clear(); 
    trLctQuality.Clear(); 
    trLctglobalPhi.Clear();   
    trLctglobalEta.Clear(); 

    trLctstripNum.Clear();   
    trLctwireGroup.Clear();

    trLctEndcap.Clear(); 
    trLctSector.Clear(); 
    trLctSubSector.Clear(); 
    trLctBx.Clear(); 
    trLctBx0.Clear(); 
   
    trLctStation.Clear(); 
    trLctRing.Clear(); 
    trLctChamber.Clear(); 
    trLctTriggerCSCID.Clear(); 
    trLctFpga.Clear();	 
    
    trLctlocalPhi.Clear(); 
    //trLctlocalPhi_bend.Clear(); 
    trLctCLCT_pattern.Clear(); 
    trLctQuality.Clear(); 
    trLctglobalPhi.Clear();   
    trLctglobalEta.Clear(); 

    trLctstripNum.Clear();   
    trLctwireGroup.Clear();

    //---------------------

    trLctEndcap.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctSector.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctSubSector.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctBx.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctBx0.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
   
    trLctStation.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctRing.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctChamber.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctTriggerCSCID.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctFpga.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR);	 
    
    trLctlocalPhi.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    //trLctlocalPhi_bend.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctCLCT_pattern.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctQuality.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 
    trLctglobalPhi.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR);   
    trLctglobalEta.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR); 

    trLctstripNum.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR);   
    trLctwireGroup.ResizeTo(MAXCSCTFTR,MAXCSCTFLCTSTR);

    //---------------------------------------------------------------------- 
    // ALL LCT 
    //---------------------------------------------------------------------- 
    lctSize = 0;
    lctEndcap.clear(); 
    lctSector.clear(); 
    lctSubSector.clear(); 
    lctBx.clear(); 
    lctBx0.clear(); 
    
    lctStation.clear(); 
    lctRing.clear(); 
    lctChamber.clear(); 
    lctTriggerCSCID.clear(); 
    lctFpga.clear();     

    // note: the SPs return them in bits 
    lctlocalPhi.clear(); 
    //lctlocalPhi_bend.clear(); 
    lctCLCT_pattern.clear(); 
    lctQuality.clear(); 
    lctglobalPhi.clear();   
    lctglobalEta.clear(); 

    lctstripNum.clear();   
    lctwireGroup.clear();   

    //--------------------------------------------------------------------------- 
    // BASIC CSCTF information 
    //--------------------------------------------------------------------------- 
    nsp = 0; // num of SPs active in the event 
    stSPslot.clear(); 
    stL1A_BXN.clear(); 
    stTrkCounter.clear(); 
    stOrbCounter.clear();

    dtSize=0;
    dtCAL.clear();
    dtFLAG.clear();
    dtBXN.clear();
    dtSector.clear();
    dtSubSector.clear();
    dtBX0.clear();
    dtPhiBend.clear();
    dtQuality.clear();
    dtPhiPacked.clear();
    }
   	       
    //----------------------------------------------------------------------
    // TRACKS 
    //----------------------------------------------------------------------
    // csctf track candidates
    int trSize;
    std::vector<int> trEndcap; 
    std::vector<int> trSector; 
    
    std::vector<int> trBx; 

    // if set to 1 track has lct from the station 
    std::vector<int> trME1ID; 
    std::vector<int> trME2ID; 
    std::vector<int> trME3ID; 
    std::vector<int> trME4ID; 
    std::vector<int> trMB1ID;     

    std::vector<int> trME1TBin;
    std::vector<int> trME2TBin;
    std::vector<int> trME3TBin;
    std::vector<int> trME4TBin;
    std::vector<int> trMB1TBin;

    std::vector<int> trOutputLink; 
  
    // some input of the PT LUTs 
    std::vector<int> trCharge; 
    std::vector<int> trChargeValid; 
    std::vector<int> trForR; 
    std::vector<int> trPhi23; 
    std::vector<int> trPhi12;   
    std::vector<int> trPhiSign;   

    // in bits... 
    std::vector<int> trEtaBit;   
    std::vector<int> trPhiBit;   
    std::vector<int> trPtBit;   
  
    // ... converted 
    std::vector<float> trEta;   
    std::vector<float> trPhi;   
    std::vector<float> trPhi_02PI; 
    std::vector<float> trPt;   

    // + useful information
    std::vector<int> trMode;
    std::vector<int> trQuality;

    //---------------------------------------------------------------------- 
    // LCT (STUBS FORMING THE TRACK)  
    //---------------------------------------------------------------------- 
    std::vector<int> trNumLCTs; // it contains the number of LCT forming a track 
       
    TMatrixD trLctEndcap; 
    TMatrixD trLctSector; 
    TMatrixD trLctSubSector; 
    TMatrixD trLctBx; 
    TMatrixD trLctBx0; 
       
    TMatrixD trLctStation; 
    TMatrixD trLctRing; 
    TMatrixD trLctChamber; 
    TMatrixD trLctTriggerCSCID; 
    TMatrixD trLctFpga;	  

     // note: the SPs return them in bits 
    TMatrixD trLctlocalPhi; 
    //TMatrixD trLctlocalPhi_bend; 
    TMatrixD trLctCLCT_pattern; 
    TMatrixD trLctQuality; 
    TMatrixD trLctglobalPhi;   
    TMatrixD trLctglobalEta; 

    TMatrixD trLctstripNum;   
    TMatrixD trLctwireGroup;
  
    //---------------------------------------------------------------------- 
    // ALL LCT 
    //---------------------------------------------------------------------- 
    int lctSize;
    std::vector<int> lctEndcap; 
    std::vector<int> lctSector; 

    std::vector<int> lctSubSector; 
    std::vector<int> lctBx; 
    std::vector<int> lctBx0; 
    std::vector<int> lctStation; 
    std::vector<int> lctRing; 
    std::vector<int> lctChamber; 
    std::vector<int> lctTriggerCSCID; 
    std::vector<int> lctFpga;     

    // note: the SPs return them in bits 
    std::vector<int> lctlocalPhi; 
    //std::vector<int> lctlocalPhi_bend; 
    std::vector<int> lctCLCT_pattern; 
    std::vector<int> lctQuality; 
    std::vector<int> lctglobalPhi;   
    std::vector<int> lctglobalEta; 
    std::vector<int> lctstripNum;   
    std::vector<int> lctwireGroup;   

    //--------------------------------------------------------------------------- 
    // BASIC CSCTF information 
    //--------------------------------------------------------------------------- 
    int nsp; // num of SPs active in the event 
    std::vector<int> stSPslot; 
    std::vector<int> stL1A_BXN; 
    std::vector<unsigned long int> stTrkCounter; 
    std::vector<unsigned long int> stOrbCounter; 

    //DT Stub Information
    int dtSize;
    std::vector<int> dtCAL;
    std::vector<int> dtFLAG;
    std::vector<int> dtBXN;
    std::vector<int> dtSector;
    std::vector<int> dtSubSector;
    std::vector<int> dtBX0;
    std::vector<int> dtPhiBend;
    std::vector<int> dtQuality;
    std::vector<int> dtPhiPacked;

  }; 
} 
#endif


