#ifndef EgammaHLTAlgos_EgammaHLTEcalIsolation_h
#define EgammaHLTAlgos_EgammaHLTEcalIsolation_h
// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTEcalIsolation
// 
/**\class EgammaHLTEcalIsolation EgammaHLTEcalIsolation.h RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTEcalIsolation.h

 Description: sum Et of all island basic clusters in cone around candidate

 Usage:
    <usage>

*/
//
// Original Author:  Monica Vazquez Acosta
//         Created:  Tue Jun 13 12:18:22 CEST 2006
// $Id$
//
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


class EgammaHLTEcalIsolation
{

   public:
    
  // default values tuned for HLT selection (only photon)
  EgammaHLTEcalIsolation(float egEcalIso_Photon_EtMin = 0.,
			 float egEcalIso_Photon_ConeSize = 0.3){
    etMinG = egEcalIso_Photon_EtMin;
    conesizeG = egEcalIso_Photon_ConeSize;
  
    /*  
    edm::LogInfo ("category") << "EgammaHLTEcalIsolation instance:"
			      << " ptMin=" << etMinG
			      << " conesize=" << conesizeG
			      << std::endl;
    */
  }
  


  virtual ~EgammaHLTEcalIsolation();
  
 private:
  
  
  float photonPtSum(const reco::Photon *photon, const reco::SuperClusterCollection& sclusters
		    , const reco::BasicClusterCollection& bclusters
		    );
   
  
  /// Get Et cut for ecal hits
  float getetMin() { return etMinG; }
  /// Get isolation cone size. 
  float getConeSize() { return conesizeG; }
  
  // ---------- member data --------------------------------
  
  // Parameters of isolation cone geometry. 
  // Photon case
  float etMinG;
  float conesizeG;
  

};


#endif
