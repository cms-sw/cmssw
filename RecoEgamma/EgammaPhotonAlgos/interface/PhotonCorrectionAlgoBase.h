#ifndef RECOEGAMMACORRECTIONALGOS_PHOTONCORRECTIONALGOBASE_H
#define RECOEGAMMACORRECTIONALGOS_PHOTONCORRECTIONALGOBASE_H
// -*- C++ -*-
//
// Package:     RecoEgammaCorrectionAlgos
// Class  :     PhotonCorrectionAlgoBase
// 
/**\class PhotonCorrectionAlgoBase PhotonCorrectionAlgoBase.h RecoEcal/RecoEgammaCorrectionAlgos/interface/PhotonCorrectionAlgoBase.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon May 29 14:58:00 CDT 2006
// $Id$
//

// system include files

// user include files
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
// forward declarations

class PhotonCorrectionAlgoBase
{

   public:
      PhotonCorrectionAlgoBase(){};

      virtual ~PhotonCorrectionAlgoBase(){};

      static float unCorrectedEnergy(const reco::SuperClusterRef& sc);

      virtual reco::Photon applyBarrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap) = 0;       

      virtual reco::Photon applyEndcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap) = 0;     
   private:


};

#endif
