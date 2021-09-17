#ifndef ElectronIDAlgo_H
#define ElectronIDAlgo_H

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class ElectronIDAlgo {
public:
  ElectronIDAlgo(){};

  virtual ~ElectronIDAlgo(){};

  //void baseSetup(const edm::ParameterSet& conf) ;
  virtual void setup(const edm::ParameterSet& conf){};
  virtual double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) { return 0.; };

protected:
  edm::InputTag reducedBarrelRecHitCollection_;
  edm::InputTag reducedEndcapRecHitCollection_;
};

#endif  // ElectronIDAlgo_H
