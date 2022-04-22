// system include
#include <iostream>

// user includes
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ParameterSet;
}

class ClusterFilter : public edm::stream::EDFilter<> {
public:
  ClusterFilter(const edm::ParameterSet&);
  ~ClusterFilter() override = default;

private:
  bool filter(edm::Event&, edm::EventSetup const&) override;
  const int nMax_;
  const edm::EDGetTokenT<edm::DetSetVector<SiStripCluster>> clustersToken_;
};

using namespace std;
using namespace edm;

ClusterFilter::ClusterFilter(const ParameterSet& pset)
    : nMax_(pset.getParameter<int>("maxClusters")),
      clustersToken_(consumes<edm::DetSetVector<SiStripCluster>>(pset.getParameter<std::string>("ClusterProducer"))) {}

bool ClusterFilter::filter(Event& e, EventSetup const& es) {
  const edm::DetSetVector<SiStripCluster>& clusters = e.get(clustersToken_);
  int size = 0;
  for (const auto& DSViter : clusters) {
    size += DSViter.data.size();
  }
  return (size < nMax_);
}
