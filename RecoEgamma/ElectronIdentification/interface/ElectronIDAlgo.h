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
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"


class ElectronIDAlgo {

public:

  ElectronIDAlgo(){};

  virtual ~ElectronIDAlgo(){};

  //void baseSetup(const edm::ParameterSet& conf) ;
  virtual void setup(const edm::ParameterSet& conf)  {};
  virtual double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) {return 0.;};

 protected:

  //EcalClusterLazyTools getClusterShape(const edm::Event&, const edm::EventSetup&);

  edm::InputTag reducedBarrelRecHitCollection_;
  edm::InputTag reducedEndcapRecHitCollection_;
};

#endif // ElectronIDAlgo_H
