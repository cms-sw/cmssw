#ifndef SubSeedGenerator_H
#define SubSeedGenerator_H

/** \class SubSeedGenerator
 *
 ************************************************************/
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"  
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"  

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include <TMath.h>

#include <Math/VectorUtil.h>
#include <Math/Point3D.h>

class SubSeedGenerator {
 public:
  
  SubSeedGenerator(const edm::ParameterSet& conf);
  ~SubSeedGenerator();

  void setupES(const edm::EventSetup& setup) {;} //FIXME: temporary
  void run(edm::Event&, const edm::EventSetup& setup, const edm::Handle<reco::SuperClusterCollection>&, reco::ElectronPixelSeedCollection&);

 private:
  edm::InputTag initialSeeds_;
  
  double dr_, deta_, dphi_, pt_;
};

#endif // SubSeedGenerator_H


