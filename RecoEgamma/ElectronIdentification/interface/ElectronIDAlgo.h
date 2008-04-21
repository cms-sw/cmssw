#ifndef ElectronIDAlgo_H
#define ElectronIDAlgo_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"

class ElectronIDAlgo {

public:

  ElectronIDAlgo(){};

  virtual ~ElectronIDAlgo(){};

  void baseSetup(const edm::ParameterSet& conf) ;
  virtual void setup(const edm::ParameterSet& conf)  {};
  virtual double result(const reco::GsfElectron*, const edm::Event&) {return 0.;};

 protected:

  const reco::ClusterShape& getClusterShape(const reco::GsfElectron*, const edm::Event&);

  edm::InputTag barrelClusterShapeAssocProducer_;
  edm::InputTag endcapClusterShapeAssocProducer_;
};

#endif // ElectronIDAlgo_H
