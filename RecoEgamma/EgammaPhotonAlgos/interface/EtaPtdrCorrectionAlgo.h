#ifndef RECOEGAMMACORRECTIONALGOS_ETAPTDRCORRECTIONALGO_H
#define RECOEGAMMACORRECTIONALGOS_ETAPTDRCORRECTIONALGO_H

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


class EtaPtdrCorrectionAlgo : public PhotonCorrectionAlgoBase
{
 public:
  EtaPtdrCorrectionAlgo(){};
  ~EtaPtdrCorrectionAlgo(){};
  
  virtual double barrelCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap);

  virtual double endcapCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap);

};

#endif
