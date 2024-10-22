#ifndef SiStripRecHitConverter_h
#define SiStripRecHitConverter_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"

class SiStripRecHitConverter : public edm::stream::EDProducer<> {
public:
  explicit SiStripRecHitConverter(const edm::ParameterSet&);
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  SiStripRecHitConverterAlgorithm recHitConverterAlgorithm;
  std::string matchedRecHitsTag, rphiRecHitsTag, stereoRecHitsTag;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterProducer;
  bool doMatching;
};
#endif
