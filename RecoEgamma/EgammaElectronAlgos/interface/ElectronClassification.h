#ifndef ElectronClassification_H
#define ElectronClassification_H

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university 
// 12/2005
// new classification numbering and showering subclasses
// golden                     =>  0
// big brem                   => 10
// narrow                     => 20
// showering nbrem 0          => 30
// showering nbrem 1          => 31
// showering nbrem 2          => 32
// showering nbrem 3          => 33
// showering nbrem 4 ou plus  => 34
// cracks                     => 40
// endcap                     => barrel + 100
// CC 08/02/2006
//===================================================================

/*! \file ElectronClassification.h
  Egamma class for classification of electrons
  Classes are: 
  BARREL
  0 = golden; 1 = big brem; 2 = narrow; 3 = showering; 4 = cracks
  ENDCAP
  10 = golden; 11 = big brem; 12 = narrow; 13 = showering
  UNDEFINED
  -1 = undefined
*/

//#include "ElectronPhoton/EgammaAnalysis/interface/EgammaCorrector.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"


class ElectronClassification
{
 public:
  
  ElectronClassification(){electronClass_=-1;}

  int getClass() const {return electronClass_;}

  virtual void correct(reco::PixelMatchGsfElectron &);

 private:

  void classify(const reco::PixelMatchGsfElectron &);

  bool isInCrack(float eta) const;

  int electronClass_;

};

#endif




