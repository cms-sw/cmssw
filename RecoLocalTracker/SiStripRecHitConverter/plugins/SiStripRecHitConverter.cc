#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitConverter.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

SiStripRecHitConverter::SiStripRecHitConverter(edm::ParameterSet const& conf)
    : recHitConverterAlgorithm(conf, consumesCollector()),
      matchedRecHitsTag(conf.getParameter<std::string>("matchedRecHits")),
      rphiRecHitsTag(conf.getParameter<std::string>("rphiRecHits")),
      stereoRecHitsTag(conf.getParameter<std::string>("stereoRecHits")),
      doMatching(conf.getParameter<bool>("doMatching")) {
  clusterProducer =
      consumes<edmNew::DetSetVector<SiStripCluster> >(conf.getParameter<edm::InputTag>("ClusterProducer"));

  produces<SiStripRecHit2DCollection>(rphiRecHitsTag);
  produces<SiStripRecHit2DCollection>(stereoRecHitsTag);
  if (doMatching) {
    produces<SiStripMatchedRecHit2DCollection>(matchedRecHitsTag);
    produces<SiStripRecHit2DCollection>(rphiRecHitsTag + "Unmatched");
    produces<SiStripRecHit2DCollection>(stereoRecHitsTag + "Unmatched");
  }
}

void SiStripRecHitConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ClusterProducer", edm::InputTag("siStripClusters"));
  desc.add<std::string>("rphiRecHits", "rphiRecHit");
  desc.add<std::string>("stereoRecHits", "stereoRecHit");
  desc.add<std::string>("matchedRecHits", "matchedRecHit");

  SiStripRecHitConverterAlgorithm::fillPSetDescription(desc);

  // unused? could be removed after parameter gets removed from HLT menu?
  desc.addOptionalUntracked<int>("VerbosityLevel");

  descriptions.addWithDefaultLabel(desc);
}

void SiStripRecHitConverter::produce(edm::Event& e, const edm::EventSetup& es) {
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;

  SiStripRecHitConverterAlgorithm::products output;
  e.getByToken(clusterProducer, clusters);
  recHitConverterAlgorithm.initialize(es);
  recHitConverterAlgorithm.run(clusters, output);
  output.shrink_to_fit();
  LogDebug("SiStripRecHitConverter") << "found\n"
                                     << output.rphi->dataSize() << "  clusters in mono detectors\n"
                                     << output.stereo->dataSize() << "  clusters in partners stereo detectors\n";

  e.put(std::move(output.rphi), rphiRecHitsTag);
  e.put(std::move(output.stereo), stereoRecHitsTag);
  if (doMatching) {
    e.put(std::move(output.matched), matchedRecHitsTag);
    e.put(std::move(output.rphiUnmatched), rphiRecHitsTag + "Unmatched");
    e.put(std::move(output.stereoUnmatched), stereoRecHitsTag + "Unmatched");
  }
}
