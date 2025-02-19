#ifndef EgammaHLTAlgos_EgammaHLTHcalIsolationDoubleCone_h
#define EgammaHLTAlgos_EgammaHLTHcalIsolationDoubleCone_h
// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTHcalIsolationDoubleCone
// 
/**\class EgammaHLTHcalIsolationDoubleCone EgammaHLTHcalIsolationDoubleCone.h RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolationDoubleCone.h

 Description: sum pt hcal hits in cone around egamma candidate but exlude central cone
                 mostly identical to EgammaHLTHcalIsolation, but
                 with an inner exclusion cone 

 Usage:
    <usage>

*/
//
// Original Author:  Monica Vazquez Acosta - CERN
//         Created:  Tue Jun 13 12:18:35 CEST 2006
// $Id: EgammaHLTHcalIsolationDoubleCone.h,v 1.2 2007/06/28 16:58:19 ghezzi Exp $
//

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
//For160 #include "Geometry/Vector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

class EgammaHLTHcalIsolationDoubleCone
{

 public:
  
  EgammaHLTHcalIsolationDoubleCone(double egHcalIso_PtMin, double egHcalIso_ConeSize, double egHcalIso_Exclusion) :
    ptMin(egHcalIso_PtMin),conesize(egHcalIso_ConeSize),exclusion(egHcalIso_Exclusion){
      /* 
	 std::cout << "EgammaHLTHcalIsolation instance:"
	 << " ptMin=" << ptMin << "|" << ptMinG
	 << " conesize="<< conesize << "|" << conesizeG
	 << std::endl;
      */
    }


  float isolPtSum(const reco::RecoCandidate* recocandidate, const HBHERecHitCollection* hbhe, const HFRecHitCollection* hf, const CaloGeometry* geometry);


  /// Get pt cut for hcal hits
  float getptMin() { return ptMin; }
  /// Get isolation cone size. 
  float getConeSize() { return conesize; }
  /// Get exclusion region
  float getExclusion() { return exclusion; }
  
 private:

  // ---------- member data --------------------------------
   // Parameters of isolation cone geometry. 
  float ptMin;
  float conesize;
  float exclusion;
  
};


#endif
