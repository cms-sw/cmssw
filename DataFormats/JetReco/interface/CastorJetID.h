#ifndef DataFormats_JetReco_interface_CastorJetID_h
#define DataFormats_JetReco_interface_CastorJetID_h

/** \class reco::CastorJetID
 *
 * \short Castor Jet ID object
 *
 * \author Salvatore Rappoccio, JHU
 *
 * \version   Original: 17-Sept-2009 by Salvatore Rappoccio
 ************************************************************/

#include "DataFormats/Common/interface/ValueMap.h"

namespace reco {
  struct CastorJetID {

    // initialize 
    CastorJetID() {
      emEnergy = 0.0;
      hadEnergy = 0.0; 
      fem = 0.0;
      width = 0.0;
      depth = 0.0;
      fhot = 0.0;
      sigmaz = 0.0;
      nTowers = 0;
      
    }

    double emEnergy;
    double hadEnergy; 
    double fem;
    double width;
    double depth;
    double fhot;
    double sigmaz;
    int nTowers;

  };

  typedef edm::ValueMap<CastorJetID>   CastorJetIDValueMap;
}

#endif
