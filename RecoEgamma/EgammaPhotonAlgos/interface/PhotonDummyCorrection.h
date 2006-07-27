#ifndef RecoEgamma_EgammaPhotonAlgos_PhotonDummyCorrection_h
#define RecoEgamma_EgammaPhotonAlgos_PhotonDummyCorrection_h
/** \class PhotonDummyCorrection
 **  
 **
 **  $Id: PhotonDummyCorrection.h,v 1.1 2006/06/27 14:05:06 nancy Exp $
 **  $Date: 2006/06/27 14:05:06 $ 
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

  void makeCorrections(reco::Photon* pho);


};

#endif
