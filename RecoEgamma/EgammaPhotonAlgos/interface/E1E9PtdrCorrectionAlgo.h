#ifndef RECOEGAMMACORRECTIONALGOS_E1E9PTDRCORRECTIONALGO_H
#define RECOEGAMMACORRECTIONALGOS_E1E9PTDRCORRECTIONALGO_H

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"




class E1E9PtdrCorrectionAlgo : public PhotonCorrectionAlgoBase
{
 public:
  E1E9PtdrCorrectionAlgo(){};
  ~E1E9PtdrCorrectionAlgo(){};
  
  virtual double barrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap);
  virtual double endcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap);

};

#endif
