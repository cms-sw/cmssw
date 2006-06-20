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
// $Id$
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



class EgammaHLTHcalIsolation
{

 public:
  
  EgammaHLTHcalIsolation(float egHcalIso_Electron_PtMin = 0., 
			float egHcalIso_Electron_ConeSize = 0.15,
			float egHcalIso_Photon_PtMin = 0.,
			float egHcalIso_Photon_ConeSize = 0.3){
    
    
    ptMin     = egHcalIso_Electron_PtMin;
    conesize  = egHcalIso_Electron_ConeSize;
    ptMinG    = egHcalIso_Photon_PtMin;
    conesizeG = egHcalIso_Photon_ConeSize;
    
    /* 
    edm::LogInfo ("category") << "EgammaHLTHcalIsolation instance:"
	      << " ptMin=" << ptMin << "|" << ptMinG
	      << " conesize="<< conesize << "|" << conesizeG
	      << std::endl;
    */
  }

  virtual ~EgammaHLTHcalIsolation();


  float electronPtSum(const reco::Electron *electron, const HBHERecHitCollection& hbhe, const HFRecHitCollection& hf, const CaloGeometry& geometry);
  float photonPtSum(const reco::Photon *photon, const HBHERecHitCollection& hbhe, const HFRecHitCollection& hf, const CaloGeometry& geometry);


  /// Get pt cut for hcal hits
  float getptMin(bool getE=true) { 
    if(getE) return ptMin; 
    else return ptMinG; }
  /// Get isolation cone size. 
  float getConeSize(bool getE=true) { 
    if(getE) return conesize; 
    else return conesizeG; }
  
 private:

  // ---------- member data --------------------------------
   // Parameters of isolation cone geometry. 
  // I Electron case
  float ptMin;
  float conesize;
  // II Photon case
  float ptMinG;
  float conesizeG;
  
};


#endif
