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
//   $Date: 2008/04/16 23:21:14 $
//   $Revision: 1.1 $ 
//
//   Original Author :
//   Hannes Sakulin      HEPHY / Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef CondFormatsL1TObjects_L1MuTriggerPtScale_h
#define CondFormatsL1TObjects_L1MuTriggerPtScale_h

#include <cmath>
#include <iostream>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/L1TObjects/interface/L1MuScale.h"

class L1MuTriggerPtScale {
 public:

  /// constructor
   L1MuTriggerPtScale() {}

   L1MuTriggerPtScale( int nbitPackingPt,
		       bool signedPackingPt,
		       int nbinsPt,
		       const std::vector<double>& scalePt
		      ) {

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
    m_PtScale = L1MuBinnedScale ( nbitPackingPt,
				  signedPackingPt,
				  nbinsPt,
				  scalePt ) ;

  };


  /// destructor
  virtual ~L1MuTriggerPtScale() {
  };
  
  /// get the Pt scale
  const L1MuScale* getPtScale() const { return &m_PtScale;};
  

 private:
  L1MuBinnedScale m_PtScale;
};


#endif
