#ifndef EgammaHLTAlgos_EgammaHLTHcalIsolation_h
#define EgammaHLTAlgos_EgammaHLTHcalIsolation_h
// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTHcalIsolation
// 
/**\class EgammaHLTHcalIsolation EgammaHLTHcalIsolation.h RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolation.h

 Description: sum pt hcal hits in cone around egamma candidate

 Usage:
    <usage>

*/
//
// Original Author:  Monica Vazquez Acosta - CERN
//         Created:  Tue Jun 13 12:18:35 CEST 2006
// $Id: EgammaHLTHcalIsolation.h,v 1.1 2006/06/20 11:27:40 monicava Exp $
//

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

class EgammaHLTHcalIsolation
{

 public:
  
  EgammaHLTHcalIsolation(double egHcalIso_PtMin, double egHcalIso_ConeSize) :
    ptMin(egHcalIso_PtMin),conesize(egHcalIso_ConeSize){
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

  
 private:

  // ---------- member data --------------------------------
   // Parameters of isolation cone geometry. 
  float ptMin;
  float conesize;
  
};


#endif
