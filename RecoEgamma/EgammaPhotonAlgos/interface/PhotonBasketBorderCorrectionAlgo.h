#ifndef EGAMMAPHOTONALGOS_PHOTONBASKETBORDERCORRECTIONALGO_H
#define EGAMMAPHOTONALGOS_PHOTONBASKETBORDERCORRECTIONALGO_H

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


class PhotonBasketBorderCorrectionAlgo : public PhotonCorrectionAlgoBase
{

 public:
  PhotonBasketBorderCorrectionAlgo(){};
  virtual ~PhotonBasketBorderCorrectionAlgo(){};
  
  virtual double barrelCorrection(const reco::Photon&ph, const reco::BasicClusterShapeAssociationCollection &clshpMap);
  virtual double endcapCorrection(const reco::Photon&ph, const reco::BasicClusterShapeAssociationCollection &clshpMap);

 private:
  
  //functions for barrel correction
  float basicClusterCorrEnergy(const reco::BasicCluster&bc,  const reco::ClusterShape& shape);
  float zeroCrackCor(float &logeta);
  float crackCorPhi(float &logphi);
  float crackCorEta(float &logeta);
};

#endif
