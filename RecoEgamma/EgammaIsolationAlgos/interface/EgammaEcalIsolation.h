#ifndef EgammaIsolationAlgos_EgammaEcalIsolation_h
#define EgammaIsolationAlgos_EgammaEcalIsolation_h

//*****************************************************************************
// File:      EgammaEcalIsolation.h
// ----------------------------------------------------------------------------
// Type:      Class implementation header
// Package:   EgammaIsolationAlgos/EgammaIsolationAlgos
// Class:     EgammaEcalIsolation
// Language:  Standard C++
// Project:   CMS
// OrigAuth:  Gilles De Lentdecker
// Institute: IIHE-ULB
//=============================================================================
//*****************************************************************************


#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


class EgammaEcalIsolation
{

   public:

  EgammaEcalIsolation(double extRadius,
		      double etLow,
		      const reco::BasicClusterCollection* ,
		      const reco::SuperClusterCollection*);


  ~EgammaEcalIsolation();
  
  double getEcalEtSum(const reco::Candidate*);
 private:
  
  // ---------- member data --------------------------------
  
  // Parameters of isolation cone geometry. 
  // Photon case
  double etMin;
  double conesize;

  const reco::BasicClusterCollection* basicClusterCollection_;
  const reco::SuperClusterCollection* superClusterCollection_;


};


#endif
