#ifndef RecoEgamma_EgammaPhotonAlgos_PhotonCorrectionAlgo_h
#define RecoEgamma_EgammaPhotonAlgos_PhotonCorrectionAlgo_h
/** \class PhotonCorrectionAlgo
 **  
 **
 **  $Id: PhotonCorrectionAlgo $
 **  $Date: $ 
 **  $Revision:  $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/EgammaCandidates/interface/Photon.h"
// C/C++ headers
#include <string>
#include <vector>
#include <set>



class PhotonCorrectionAlgo {

 public:
  

  PhotonCorrectionAlgo();
  
  virtual ~PhotonCorrectionAlgo(){}

  void makeCorrections(reco::Photon* pho);


};

#endif
