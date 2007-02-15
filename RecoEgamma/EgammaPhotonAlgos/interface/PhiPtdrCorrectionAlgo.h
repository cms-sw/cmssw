#ifndef RECOEGAMMACORRECTIONALGOS_PHIPTDRCORRECTIONALGO_H
#define RECOEGAMMACORRECTIONALGOS_PHIPTDRCORRECTIONALGO_H

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


class PhiPtdrCorrectionAlgo : public PhotonCorrectionAlgoBase
{
 public:
  PhiPtdrCorrectionAlgo(){};
  ~PhiPtdrCorrectionAlgo(){};
  
  virtual double barrelCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap);

  virtual double endcapCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap);

};

#endif
