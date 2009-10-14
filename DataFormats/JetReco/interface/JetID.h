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
    double fHPD; 
    double fRBX;
    double n90Hits;
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
  };

  typedef edm::ValueMap<JetID>   JetIDValueMap;
}

#endif
