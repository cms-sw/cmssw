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
// $Id: PhotonCorrectionAlgoBase.h,v 1.1 2006/12/17 14:49:43 futyand Exp $
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

      virtual double barrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap) = 0;       

      virtual double endcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap) = 0;     
   private:


};

#endif
