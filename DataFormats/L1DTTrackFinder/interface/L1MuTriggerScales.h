//-------------------------------------------------
//
//   \class L1MuTriggerScales
//
/**   Description:  Class that creates all scales used to pass 
 *                  data from the regional muon triggers to
 *                  the Global Muon Trigger and from the latter 
 *                  to the Global Trigger
*/                  
//                
//   $Date: 2006/06/01 00:00:00 $
//   $Revision: 1.1 $ 
//
//   Author :
//   Hannes Sakulin      HEPHY / Vienna
//
//--------------------------------------------------
#ifndef L1MU_TRIGGER_SCALES_H
#define L1MU_TRIGGER_SCALES_H

using namespace std;

#include <cmath>
#include <iostream>

#include "DataFormats/L1DTTrackFinder/interface/L1MuScale.h"

class L1MuTriggerScales {
 public:

  /// constructor
  L1MuTriggerScales() {

    //
    // Regional Muon Trigger Eta Scales
    //
    const float rpcetabins[34]= {
      -2.10, -1.97, -1.85, -1.73, -1.61, -1.48,
      -1.36, -1.24, -1.14, -1.04, -0.93, -0.83, 
      -0.72, -0.58, -0.44, -0.27, -0.07,     
              0.07,  0.27,  0.44,  0.58,  0.72,
       0.83,  0.93,  1.04,  1.14,  1.24,  1.36,
       1.48,  1.61,  1.73,  1.85,  1.97,  2.10};
    
    m_RegionalEtaScale[0] = new L1MuBinnedScale<L1MuSignedPacking<6> > (64, -1.2, 1.2, 32); // DT

    // RPC index -16 .. 16
    m_RegionalEtaScale[1] = new L1MuBinnedScale<L1MuSignedPacking<6> > (33, rpcetabins, 16); // brl RPC

    m_RegionalEtaScale[2] = new L1MuSymmetricBinnedScale<6> (32, 0.9, 2.5);    // CSC

    // RPC index -16 .. 16
    m_RegionalEtaScale[3] = new L1MuBinnedScale<L1MuSignedPacking<6> > (33, rpcetabins, 16); // fwd RPC

    //
    // Eta scale at GMT output
    //

    const float gmt_outputetascale[32] = {  0.00,
            0.10,  0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80, 
            0.90,  1.00,  1.10,  1.20,  1.30,  1.40,  1.50,  1.60,
            1.70,  1.75,  1.80,  1.85,  1.90,  1.95,  2.00,  2.05,
            2.10,  2.15,  2.20,  2.25,  2.30,  2.35,  2.40 };

    m_GMTEtaScale = new L1MuSymmetricBinnedScale<6> (31, gmt_outputetascale);

    //
    // Phi Scale. Common to all Regioanl Muon Triggers and GMT
    // 

    m_PhiScale = new L1MuBinnedScale<L1MuUnsignedPacking<8> >(144, 0., 2. * M_PI);

    //
    // Pt Scale. Common to all Regioanl Muon Triggers and GMT
    // 

    // pt scale in GeV
    // low edges of pt bins
    const float ptscale[33] = { 
      -1.,   0.0,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0,
      4.5,   5.0,   6.0,   7.0,   8.0,  10.0,  12.0,  14.0,  
      16.0,  18.0,  20.0,  25.0,  30.0,  35.0,  40.0,  45.0, 
      50.0,  60.0,  70.0,  80.0,  90.0, 100.0, 120.0, 140.0, 1.E6 };

    m_PtScale = new L1MuBinnedScale<L1MuUnsignedPacking<5> >(32, ptscale) ;
    
  };

  /// destructor
  virtual ~L1MuTriggerScales() {
    for (int i=0; i<4; i++) 
      delete m_RegionalEtaScale[i];

    delete m_GMTEtaScale;
    delete m_PhiScale;
    delete m_PtScale; 
  };
  
  /// get the regioanl muon trigger eta scale, isys = 0(DT), 1(bRPC), 2(CSC), 3(fwdRPC)
  L1MuScale* getRegionalEtaScale(int isys) const { 
    if (isys<0 || isys>3) cout << "Error in L1MuTriggerScales:: isys out of range: " << isys << endl; 
    return m_RegionalEtaScale[isys]; 
  };

  /// get the GMT eta scale
  L1MuScale* getGMTEtaScale() const { return m_GMTEtaScale; };

  /// get the phi scale
  L1MuScale* getPhiScale() const { return m_PhiScale;};
  
  /// get the Pt scale
  L1MuScale* getPtScale() const { return m_PtScale;};
  

 private:
  L1MuScale *m_RegionalEtaScale[4];
  L1MuScale *m_GMTEtaScale;
  L1MuScale *m_PhiScale;
  L1MuScale *m_PtScale;
};


#endif









































