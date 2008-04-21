#ifndef GlobalSeedProducer_h
#define GlobalSeedProducer_h
  
//
// Package:         RecoEgamma/EgammaElectronProducers
// Class:           GlobalSeedProducer
// 
// Description:     Calls SubSeedGenerator
//                  to find TrackingSeeds.
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
 
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class  SubSeedGenerator;

class GlobalSeedProducer : public edm::EDProducer
{
 public:
  
  explicit GlobalSeedProducer(const edm::ParameterSet& conf);
  
  virtual ~GlobalSeedProducer();
  
  virtual void beginJob(edm::EventSetup const&iSetup);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:


  edm::InputTag superClusters_[2];
  
  const edm::ParameterSet conf_;
  SubSeedGenerator *matcher_;
 
};
  
#endif
 


