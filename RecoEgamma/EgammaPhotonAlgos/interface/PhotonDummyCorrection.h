#ifndef RecoEgamma_EgammaPhotonAlgos_PhotonDummyCorrection_h
#define RecoEgamma_EgammaPhotonAlgos_PhotonDummyCorrection_h
/** \class PhotonDummyCorrection
 **  
 **
 **  $Id: PhotonDummyCorrection.h,v 1.1 2006/07/27 19:35:11 nancy Exp $
 **  $Date: 2006/07/27 19:35:11 $ 
 **  $Revision: 1.1 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/EgammaCandidates/interface/Photon.h"
// C/C++ headers
#include <string>
#include <vector>
#include <set>



class PhotonDummyCorrection {

 public:
  

  PhotonDummyCorrection();
  
  virtual ~PhotonDummyCorrection(){}

  reco::Photon makeCorrections(const reco::Photon& pho);


};

#endif
