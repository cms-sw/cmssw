#ifndef RECOEGAMMACORRECTIONALGOS_E1E9CORRECTIONALGO_H
#define RECOEGAMMACORRECTIONALGOS_E1E9CORRECTIONALGO_H

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"




class E1E9CorrectionAlgo : public PhotonCorrectionAlgoBase
{
 public:
  E1E9CorrectionAlgo(){};
  ~E1E9CorrectionAlgo(){};
  
  reco::Photon applyBarrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap);
  reco::Photon applyEndcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap);

};

#endif
