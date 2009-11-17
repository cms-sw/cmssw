#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTPixelActivityFilter : public HLTFilter {
public:
  explicit HLTPixelActivityFilter(const edm::ParameterSet&);
  ~HLTPixelActivityFilter();
  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:
  edm::InputTag pixelTag_;          // input tag identifying product containing pixel clusters
  unsigned int  min_clusters_;      // minimum number of clusters
};

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
 
HLTPixelActivityFilter::HLTPixelActivityFilter(const edm::ParameterSet& iConfig) :
  pixelTag_     (iConfig.getParameter<edm::InputTag>("pixelTag")),
  min_clusters_ (iConfig.getParameter<unsigned int>("minClusters"))
{
  LogDebug("") << "Using the " << pixelTag_ << " input collection";
  LogDebug("") << "Requesting " << min_clusters_ << " clusters";
}

HLTPixelActivityFilter::~HLTPixelActivityFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTPixelActivityFilter::filter(edm::Event& event, const edm::EventSetup& iSetup)
{
  // get hold of products from Event
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusterColl;
  event.getByLabel(pixelTag_,clusterColl);

  unsigned int clusterSize = clusterColl->size();
  bool accept = (clusterSize >= min_clusters_);

  LogDebug("") << "Number of clusters accepted: " << clusterSize;

  // return with final filter decision
  return accept;
}
