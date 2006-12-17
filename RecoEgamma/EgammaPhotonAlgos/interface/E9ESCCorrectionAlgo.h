#ifndef RECOEGAMMACORRECTIONALGOS_E9ESCCORRECTIONALGO_H
#define RECOEGAMMACORRECTIONALGOS_E9ESCCORRECTIONALGO_H

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class E9ESCCorrectionAlgo : public PhotonCorrectionAlgoBase
{
 public:
  E9ESCCorrectionAlgo(){};
  ~E9ESCCorrectionAlgo(){};
  
  virtual reco::Photon applyBarrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap) ;      

  virtual reco::Photon applyEndcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap) ;     
};

#endif
