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
//   $Date: 2009/05/22 12:19:03 $
//   $Revision: 1.6 $ 
//
//   Original Author :
//   Hannes Sakulin      HEPHY / Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef CondFormatsL1TObjects_L1MuTriggerScales_h
#define CondFormatsL1TObjects_L1MuTriggerScales_h

#include <cmath>
#include <iostream>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/L1TObjects/interface/L1MuScale.h"

class L1MuTriggerScales {
 public:

  /// constructor
   L1MuTriggerScales() {}

   L1MuTriggerScales( int nbitPackingDTEta,
		      bool signedPackingDTEta,
		      int nbinsDTEta,
		      float minDTEta,
		      float maxDTEta,
		      int offsetDTEta,

		      int nbitPackingCSCEta,
		      int nbinsCSCEta,
		      float minCSCEta,
		      float maxCSCEta,

		      const std::vector<double>& scaleRPCEta,
		      int nbitPackingBrlRPCEta,
		      bool signedPackingBrlRPCEta,
		      int nbinsBrlRPCEta,
		      int offsetBrlRPCEta,
		      int nbitPackingFwdRPCEta,
		      bool signedPackingFwdRPCEta,
		      int nbinsFwdRPCEta,
		      int offsetFwdRPCEta,

		      int nbitPackingGMTEta,
		      int nbinsGMTEta,
		      const std::vector<double>& scaleGMTEta,

		      int nbitPackingPhi,
		      bool signedPackingPhi,
		      int nbinsPhi,
		      float minPhi,
		      float maxPhi

/* 		      int nbitPackingPt, */
/* 		      bool signedPackingPt, */
/* 		      int nbinsPt, */
/* 		      const std::vector<double>& scalePt */

		      ) {

    //
    // Regional Muon Trigger Eta Scales
    //
/*     const float rpcetabins[34]= { */
/*       -2.10, -1.97, -1.85, -1.73, -1.61, -1.48, */
/*       -1.36, -1.24, -1.14, -1.04, -0.93, -0.83,  */
/*       -0.72, -0.58, -0.44, -0.27, -0.07,      */
/*               0.07,  0.27,  0.44,  0.58,  0.72, */
/*        0.83,  0.93,  1.04,  1.14,  1.24,  1.36, */
/*        1.48,  1.61,  1.73,  1.85,  1.97,  2.10}; */

    // DT
    //m_RegionalEtaScale[0] = L1MuBinnedScale( 6, true, 64, -1.2, 1.2, 32);
    m_RegionalEtaScale[0] = L1MuBinnedScale( nbitPackingDTEta,
					     signedPackingDTEta,
					     nbinsDTEta,
					     minDTEta,
					     maxDTEta,
					     offsetDTEta );

    // RPC index -16 .. 16, brl RPC
    // m_RegionalEtaScale[1] = L1MuBinnedScale (6, true, 33, rpcetabins, 16);
    m_RegionalEtaScale[1] = L1MuBinnedScale (nbitPackingBrlRPCEta,
					     signedPackingBrlRPCEta,
					     nbinsBrlRPCEta,
					     scaleRPCEta,
					     offsetBrlRPCEta ) ;

    // CSC
    m_RegionalEtaScale[2] = L1MuBinnedScale() ;
    // // m_RegionalEtaScale[2] = L1MuSymmetricBinnedScale ( 6, 32, 0.9, 2.5);
    //    m_RegionalEtaScaleCSC = L1MuSymmetricBinnedScale ( 6, 32, 0.9, 2.5);
    m_RegionalEtaScaleCSC = L1MuSymmetricBinnedScale ( nbitPackingCSCEta,
						       nbinsCSCEta,
						       minCSCEta,
						       maxCSCEta );

    // RPC index -16 .. 16, fwd RPC
    // m_RegionalEtaScale[3] = L1MuBinnedScale (6, true, 33, rpcetabins, 16);
    m_RegionalEtaScale[3] = L1MuBinnedScale (nbitPackingFwdRPCEta,
					     signedPackingFwdRPCEta,
					     nbinsFwdRPCEta,
					     scaleRPCEta,
					     offsetFwdRPCEta );

    //
    // Eta scale at GMT output
    //

/*     const float gmt_outputetascale[32] = {  0.00, */
/*             0.10,  0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80,  */
/*             0.90,  1.00,  1.10,  1.20,  1.30,  1.40,  1.50,  1.60, */
/*             1.70,  1.75,  1.80,  1.85,  1.90,  1.95,  2.00,  2.05, */
/*             2.10,  2.15,  2.20,  2.25,  2.30,  2.35,  2.40 }; */

    // m_GMTEtaScale = L1MuSymmetricBinnedScale (6, 31, gmt_outputetascale);
    m_GMTEtaScale = L1MuSymmetricBinnedScale (nbitPackingGMTEta,
					      nbinsGMTEta,
					      scaleGMTEta );

    //
    // Phi Scale. Common to all Regioanl Muon Triggers and GMT
    // 

    // m_PhiScale = L1MuBinnedScale (8, false, 144, 0., 2. * M_PI);
    m_PhiScale = L1MuBinnedScale (nbitPackingPhi,
				  signedPackingPhi,
				  nbinsPhi,
				  minPhi,
				  maxPhi );

    //
    // Pt Scale. Common to all Regioanl Muon Triggers and GMT
    // 

    // pt scale in GeV
    // low edges of pt bins
/*     const float ptscale[33] = {  */
/*       -1.,   0.0,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0, */
/*       4.5,   5.0,   6.0,   7.0,   8.0,  10.0,  12.0,  14.0,   */
/*       16.0,  18.0,  20.0,  25.0,  30.0,  35.0,  40.0,  45.0,  */
/*       50.0,  60.0,  70.0,  80.0,  90.0, 100.0, 120.0, 140.0, 1.E6 }; */

    // m_PtScale = L1MuBinnedScale ( 5, false, 32, ptscale) ;
/*     m_PtScale = L1MuBinnedScale ( nbitPackingPt, */
/* 				  signedPackingPt, */
/* 				  nbinsPt, */
/* 				  scalePt ) ; */

  };


  /// destructor
  virtual ~L1MuTriggerScales() {
//     for (int i=0; i<4; i++) 
//       delete m_RegionalEtaScale[i];

//     delete m_GMTEtaScale;
//     delete m_PhiScale;
//     delete m_PtScale; 
  };
  
  /// get the regioanl muon trigger eta scale, isys = 0(DT), 1(bRPC), 2(CSC), 3(fwdRPC)
  const L1MuScale* getRegionalEtaScale(int isys) const { 
    if (isys<0 || isys>3) edm::LogWarning("ScaleRangeViolation") << "Error in L1MuTriggerScales:: isys out of range: " << isys;
    if( isys == 2 )
    {
       return &m_RegionalEtaScaleCSC ;
    }
    else
    {
       return &( m_RegionalEtaScale[isys] ); 
    }
  };

  /// get the GMT eta scale
  const L1MuScale* getGMTEtaScale() const { return &m_GMTEtaScale ; };

  /// set the GMT eta scale
  void setGMTEtaScale(const L1MuSymmetricBinnedScale& scale)  { m_GMTEtaScale = scale ; };


  /// get the phi scale
  const L1MuScale* getPhiScale() const { return &m_PhiScale;};

  /// set the phi scale
  void setPhiScale(const L1MuBinnedScale& scale)  { m_PhiScale = scale ; };
  
/*   /// get the Pt scale */
/*   const L1MuScale* getPtScale() const { return &m_PtScale;}; */
  

 private:
  L1MuBinnedScale m_RegionalEtaScale[4]; // 2=csc will be empty
  L1MuSymmetricBinnedScale m_RegionalEtaScaleCSC ;
  L1MuSymmetricBinnedScale m_GMTEtaScale;
  L1MuBinnedScale m_PhiScale;
  //  L1MuBinnedScale m_PtScale;
};


#endif
