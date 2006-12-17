#ifndef RECOEGAMMACORRECTIONALGOS_ETACORRECTIONALGO_H
#define RECOEGAMMACORRECTIONALGOS_ETACORRECTIONALGO_H

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


class EtaCorrectionAlgo : public PhotonCorrectionAlgoBase
{
 public:
  EtaCorrectionAlgo(){};
  ~EtaCorrectionAlgo(){};
  
  virtual reco::Photon applyBarrelCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap);

  virtual reco::Photon applyEndcapCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap);

};

#endif
