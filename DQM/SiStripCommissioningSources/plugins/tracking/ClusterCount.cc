#include "DQM/SiStripCommissioningSources/plugins/tracking/ClusterCount.h"

//
// constructors and destructor
//
ClusterCount::ClusterCount(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  //   clusterLabel_ = iConfig.getParameter<edm::InputTag>("ClustersLabel");
  clusterToken_ = consumes<edm::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("ClustersLabel"));
}

ClusterCount::~ClusterCount() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void ClusterCount::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {}

//
// member functions
//

// ------------ method called to for each event  ------------
void ClusterCount::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  // look at the clusters
  edm::Handle<edm::DetSetVector<SiStripCluster> > clusters;
  iEvent.getByToken(clusterToken_, clusters);
  const edm::DetSetVector<SiStripCluster>* clusterSet = clusters.product();
  // loop on the detsetvector<cluster>
  for (auto DSViter = clusterSet->begin(); DSViter != clusterSet->end(); DSViter++) {
    auto begin = DSViter->data.begin();
    auto end = DSViter->data.end();
    for (auto iter = begin; iter != end; ++iter) {
      LogDebug("ReconstructedClusters") << "Detid/Strip: " << std::hex << DSViter->id << std::dec << " / "
                                        << iter->barycenter();
    }
  }
}
