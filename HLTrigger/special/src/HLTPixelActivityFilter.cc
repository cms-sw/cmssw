#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTPixelActivityFilter : public HLTFilter {
public:
  explicit HLTPixelActivityFilter(const edm::ParameterSet&);
  ~HLTPixelActivityFilter();

private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);

  edm::InputTag inputTag_;          // input tag identifying product containing pixel clusters
  bool          saveTags_;           // whether to save this tag
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
 
HLTPixelActivityFilter::HLTPixelActivityFilter(const edm::ParameterSet& config) :
  inputTag_     (config.getParameter<edm::InputTag>("inputTag")),
  saveTags_      (config.getParameter<bool>("saveTags")),
  min_clusters_ (config.getParameter<unsigned int>("minClusters")),
  max_clusters_ (config.getParameter<unsigned int>("maxClusters"))
{
  LogDebug("") << "Using the " << inputTag_ << " input collection";
  LogDebug("") << "Requesting at least " << min_clusters_ << " clusters";
  if(max_clusters_ > 0) 
    LogDebug("") << "...but no more than " << max_clusters_ << " clusters";

  // register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
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
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTags_) filterobject->addCollectionTag(inputTag_);

  // get hold of products from Event
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusterColl;
  event.getByLabel(inputTag_, clusterColl);

  unsigned int clusterSize = clusterColl->dataSize();
  LogDebug("") << "Number of clusters accepted: " << clusterSize;
  bool accept = (clusterSize >= min_clusters_);
  if(max_clusters_ > 0) 
    accept &= (clusterSize <= max_clusters_);

  // put filter object into the Event
  event.put(filterobject);

  // return with final filter decision
  return accept;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPixelActivityFilter);
