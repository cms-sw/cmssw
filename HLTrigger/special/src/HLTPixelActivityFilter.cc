#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTPixelActivityFilter : public HLTFilter {
public:
  explicit HLTPixelActivityFilter(const edm::ParameterSet&);
  ~HLTPixelActivityFilter();

private:
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

  edm::InputTag inputTag_;          // input tag identifying product containing pixel clusters
  unsigned int  min_clusters_;      // minimum number of clusters
  unsigned int  max_clusters_;      // maximum number of clusters

};

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

//
// constructors and destructor
//
 
HLTPixelActivityFilter::HLTPixelActivityFilter(const edm::ParameterSet& config) : HLTFilter(config),
  inputTag_     (config.getParameter<edm::InputTag>("inputTag")),
  min_clusters_ (config.getParameter<unsigned int>("minClusters")),
  max_clusters_ (config.getParameter<unsigned int>("maxClusters"))
{
  LogDebug("") << "Using the " << inputTag_ << " input collection";
  LogDebug("") << "Requesting at least " << min_clusters_ << " clusters";
  if(max_clusters_ > 0) 
    LogDebug("") << "...but no more than " << max_clusters_ << " clusters";
}

HLTPixelActivityFilter::~HLTPixelActivityFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTPixelActivityFilter::hltFilter(edm::Event& event, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);

  // get hold of products from Event
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusterColl;
  event.getByLabel(inputTag_, clusterColl);

  unsigned int clusterSize = clusterColl->dataSize();
  LogDebug("") << "Number of clusters accepted: " << clusterSize;
  bool accept = (clusterSize >= min_clusters_);
  if(max_clusters_ > 0) 
    accept &= (clusterSize <= max_clusters_);

  // return with final filter decision
  return accept;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPixelActivityFilter);
