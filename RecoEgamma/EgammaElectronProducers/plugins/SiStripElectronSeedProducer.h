#ifndef SiStripElectronSeedProducer_h
#define SiStripElectronSeedProducer_h


#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripElectronSeedGenerator;

class SiStripElectronSeedProducer : public edm::global::EDProducer<>
{
 public:

  explicit SiStripElectronSeedProducer(const edm::ParameterSet& conf);

  ~SiStripElectronSeedProducer() override;

  void produce(edm::StreamID sid, edm::Event& e, const edm::EventSetup& c) const override;

 private:
  edm::EDGetTokenT<reco::SuperClusterCollection> superClusters_[2];
  edm::ParameterSet conf_;
  SiStripElectronSeedGenerator *matcher_;
  };

#endif
