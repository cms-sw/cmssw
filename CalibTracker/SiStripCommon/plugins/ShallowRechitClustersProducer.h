#ifndef SHALLOW_RECHITCLUSTERS_PRODUCER
#define SHALLOW_RECHITCLUSTERS_PRODUCER

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class ShallowRechitClustersProducer : public edm::stream::EDProducer<> {
public:
  explicit ShallowRechitClustersProducer(const edm::ParameterSet&);

private:
  std::string Suffix;
  std::string Prefix;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusters_token_;
  std::vector<edm::EDGetTokenT<SiStripRecHit2DCollection> > rec_hits_tokens_;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
};

#endif
