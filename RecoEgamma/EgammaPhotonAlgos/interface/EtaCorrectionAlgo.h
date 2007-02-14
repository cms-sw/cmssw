#ifndef RECOEGAMMACORRECTIONALGOS_ETACORRECTIONALGO_H
#define RECOEGAMMACORRECTIONALGOS_ETACORRECTIONALGO_H

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


class EtaCorrectionAlgo : public PhotonCorrectionAlgoBase
{
 public:
  EtaCorrectionAlgo(){};
  ~EtaCorrectionAlgo(){};
  
  virtual double barrelCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap);

  virtual double endcapCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap);

};

#endif
