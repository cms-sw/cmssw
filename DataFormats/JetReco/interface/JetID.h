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
    }

    // hcal+ecal id
    double fHPD; 
    double fRBX;
    int    n90Hits;
    double fSubDetector1;
    double fSubDetector2;
    double fSubDetector3;
    double fSubDetector4;
    double restrictedEMF;
    int    nHCALTowers;
    int    nECALTowers;
    double approximatefHPD;
    double approximatefRBX;
    int    hitsInN90;
    // muon hits id
    int numberOfHits2RPC;
    int numberOfHits3RPC;
    int numberOfHitsRPC;

  };

  typedef edm::ValueMap<JetID>   JetIDValueMap;
}

#endif
