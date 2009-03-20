#ifndef ElectronClassification_H
#define ElectronClassification_H

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
// 12/2005
// See GsfElectron::Classification
//===================================================================

//#include "ElectronPhoton/EgammaAnalysis/interface/EgammaCorrector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"


class ElectronClassification
{
 public:

  ElectronClassification(){electronClass_=reco::GsfElectron::UNKNOWN;}

  reco::GsfElectron::Classification getClass() const {return electronClass_;}

  virtual void correct(reco::GsfElectron &);

 private:

  void classify(const reco::GsfElectron &);

//  bool isInCrack(float eta) const;
//  bool isInEtaGaps(float eta) const;
//  bool isInPhiGaps(float phi) const;

  reco::GsfElectron::Classification electronClass_;

};

#endif




