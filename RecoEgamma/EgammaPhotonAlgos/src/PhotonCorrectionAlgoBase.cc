// -*- C++ -*-
//
// Package:     RecoEgammaCorrectionAlgos
// Class  :     PhotonCandidateCorrectionAlgoBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Mon May 29 15:09:48 CDT 2006
// $Id$
//

// system include files

// user include files
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <map>

//
// constructors and destructor
//

float PhotonCorrectionAlgoBase::unCorrectedEnergy(const reco::SuperClusterRef& sc)
{
  float sumEnergy(0);
  reco::basicCluster_iterator bcItr;
  for(bcItr = sc->clustersBegin(); bcItr != sc->clustersEnd(); bcItr++)
    {
      sumEnergy += (*bcItr)->energy();
    }

  return sumEnergy;
}


