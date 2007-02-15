#ifndef RECOEGAMMACORRECTIONALGOS_E9ESCPTDRCORRECTIONALGO_H
#define RECOEGAMMACORRECTIONALGOS_E9ESCPTDRCORRECTIONALGO_H

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class E9ESCPtdrCorrectionAlgo : public PhotonCorrectionAlgoBase
{
 public:
  E9ESCPtdrCorrectionAlgo(){};
  ~E9ESCPtdrCorrectionAlgo(){};
  
  virtual double barrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap) ;      

  virtual double endcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap) ;     
};

#endif
