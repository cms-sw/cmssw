#ifndef EgammaHLTAlgos_EgammaHLTEcalIsolation_h
#define EgammaHLTAlgos_EgammaHLTEcalIsolation_h
// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTEcalIsolation
// 
/**\class EgammaHLTEcalIsolation EgammaHLTEcalIsolation.h RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTEcalIsolation.h

 Description: sum Et of all island basic clusters in cone around candidate

*/
//
// Original Author:  Monica Vazquez Acosta
//         Created:  Tue Jun 13 12:18:22 CEST 2006
// $Id: EgammaHLTEcalIsolation.h,v 1.4 2013/05/30 21:48:56 gartung Exp $
//
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


class EgammaHLTEcalIsolation
{

   public:

  //EgammaHLTEcalIsolation(float egEcalIso_Photon_EtMin = 0., float egEcalIso_Photon_ConeSize = 0.3) : 
  EgammaHLTEcalIsolation(double egEcalIso_EtMin, double egEcalIso_ConeSize, int SC_algo_type) : 
    etMin(egEcalIso_EtMin), conesize(egEcalIso_ConeSize), algoType_(SC_algo_type) {
      /*
      std::cout << "EgammaHLTEcalIsolation instance:"
      << " ptMin=" << etMin
      << " conesize=" << conesize
      << std::endl;
      */

    }
  
  float isolPtSum(const reco::RecoCandidate *recocandidate, 
		  const std::vector<const reco::SuperCluster*>& sclusters,
		  const std::vector<const reco::BasicCluster*>& bclusters);

  /// Get Et cut for ecal hits
  float getetMin() { return etMin; }
  /// Get isolation cone size. 
  float getConeSize() { return conesize; }

 private:
  
  // ---------- member data --------------------------------
  
  // Parameters of isolation cone geometry. 
  // Photon case
  double etMin;
  double conesize;
  int algoType_;

};


#endif
