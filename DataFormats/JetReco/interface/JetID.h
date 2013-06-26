#ifndef DataFormats_JetReco_interface_JetID_h
#define DataFormats_JetReco_interface_JetID_h

/** \class reco::JetID
 *
 * \short Jet ID object
 *
 * \author Salvatore Rappoccio, JHU
 *
 * \version   Original: 17-Sept-2009 by Salvatore Rappoccio
 ************************************************************/

#include "DataFormats/Common/interface/ValueMap.h"

namespace reco {
  struct JetID {

    // initialize 
    JetID() {
      fHPD= 0.0; 
      fRBX= 0.0;
      n90Hits= 0;
      fSubDetector1= 0.0;
      fSubDetector2= 0.0;
      fSubDetector3= 0.0;
      fSubDetector4= 0.0;
      restrictedEMF= 0.0;
      nHCALTowers= 0;
      nECALTowers= 0;
      approximatefHPD= 0.0;
      approximatefRBX= 0.0;
      hitsInN90= 0;
      numberOfHits2RPC= 0;
      numberOfHits3RPC= 0;
      numberOfHitsRPC= 0;
      
      fEB = fEE = fHB = fHE = fHO = fLong = fShort = 0.0;
      fLS = fHFOOT = 0.0;
      
    }



    // hcal+ecal id
    float fHPD; 
    float fRBX;
    short    n90Hits;
    float fSubDetector1;
    float fSubDetector2;
    float fSubDetector3;
    float fSubDetector4;
    float restrictedEMF;
    short    nHCALTowers;
    short    nECALTowers;
    float approximatefHPD;
    float approximatefRBX;
    short    hitsInN90;
    // muon hits id
    short numberOfHits2RPC;
    short numberOfHits3RPC;
    short numberOfHitsRPC;
    
    float fEB, fEE, fHB, fHE, fHO, fLong, fShort;
    float fLS, fHFOOT;

  };

  typedef edm::ValueMap<JetID>   JetIDValueMap;
}

#endif
