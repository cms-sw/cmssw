#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

//
// class declaration
//

class HLTPixelActivityFilter : public HLTFilter {
public:
  explicit HLTPixelActivityFilter(const edm::ParameterSet&);
  ~HLTPixelActivityFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
//  int countLayersWithClusters(edm::Handle<edmNew::DetSetVector<SiPixelCluster> > & clusterCol,const TrackerTopology& tTopo);

  edm::InputTag inputTag_;          // input tag identifying product containing pixel clusters
  unsigned int  min_clusters_;      // minimum number of clusters
  unsigned int  max_clusters_;      // maximum number of clusters
  unsigned int  min_layers_;      // minimum number of clusters
  unsigned int  max_layers_;      // maximum number of clusters
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > inputToken_;

};

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
//
// constructors and destructor
//

HLTPixelActivityFilter::HLTPixelActivityFilter(const edm::ParameterSet& config) : HLTFilter(config),
  inputTag_     (config.getParameter<edm::InputTag>("inputTag")),
  min_clusters_ (config.getParameter<unsigned int>("minClusters")),
  max_clusters_ (config.getParameter<unsigned int>("maxClusters")),
  min_layers_ (config.getParameter<unsigned int>("minLayers")),
  max_layers_ (config.getParameter<unsigned int>("maxLayers"))
{
  inputToken_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(inputTag_);
  LogDebug("") << "Using the " << inputTag_ << " input collection";
  LogDebug("") << "Requesting at least " << min_clusters_ << " clusters";
  if(max_clusters_ > 0)
    LogDebug("") << "...but no more than " << max_clusters_ << " clusters";
  if(min_layers_ > 0)
    LogDebug("") << "Also requesting at least " << min_layers_ << " layers with clusters";
  if(max_layers_ > 0)
    LogDebug("") << "...but no more than " << max_layers_ << " layers with clusters";

}

HLTPixelActivityFilter::~HLTPixelActivityFilter() = default;

void
HLTPixelActivityFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltSiPixelClusters"));
  desc.add<unsigned int>("minClusters",3);
  desc.add<unsigned int>("maxClusters",0);
  desc.add<unsigned int>("minLayers",0);
  desc.add<unsigned int>("maxLayers",0);
  descriptions.add("hltPixelActivityFilter",desc);
}

//
// member functions
//
// ------------ method called to produce the data  ------------
bool HLTPixelActivityFilter::hltFilter(edm::Event& event, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);

  // get hold of products from Event
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusterColl;
  event.getByToken(inputToken_, clusterColl);

  unsigned int clusterSize = clusterColl->dataSize();
  LogDebug("") << "Number of clusters accepted: " << clusterSize;
  bool accept = (clusterSize >= min_clusters_);
  if(max_clusters_ > 0)
    accept &= (clusterSize <= max_clusters_);

  if (min_layers_ > 0 || max_layers_ > 0){

	  edm::ESHandle<TrackerTopology> tTopoHandle;
	  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
	  const TrackerTopology& tTopo = *tTopoHandle;
	  unsigned int layerCount = 0;
	  const edmNew::DetSetVector<SiPixelCluster>& clusters = *clusterColl;
	    
	  edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=clusters.begin();
	   

	  std::vector<int> foundLayersB;
	  std::vector<int> foundLayersEp;
	  std::vector<int> foundLayersEm;  
	  for ( ; DSViter != clusters.end() ; DSViter++) {
		unsigned int detid = DSViter->detId();
		DetId detIdObject( detid );
		const auto nCluster = DSViter->size();	
		const auto subdet = detIdObject.subdetId();
		if (subdet == PixelSubdetector::PixelBarrel){
			if(!(std::find(foundLayersB.begin(), foundLayersB.end(), tTopo.layer(detIdObject)) != foundLayersB.end()) && nCluster > 0) foundLayersB.push_back(tTopo.layer(detIdObject));
		}
		else if (subdet ==PixelSubdetector::PixelEndcap){
			if (tTopo.side(detIdObject) == 2){
				if(!(std::find(foundLayersEp.begin(), foundLayersEp.end(), tTopo.layer(detIdObject)) != foundLayersEp.end()) && nCluster > 0) foundLayersEp.push_back(tTopo.layer(detIdObject));
			}
			else if (tTopo.side(detIdObject) == 1){
				if(!(std::find(foundLayersEm.begin(), foundLayersEm.end(), tTopo.layer(detIdObject)) != foundLayersEm.end()) && nCluster > 0) foundLayersEm.push_back(tTopo.layer(detIdObject));
			}

		}
	  }
	  layerCount = foundLayersB.size()+foundLayersEp.size()+foundLayersEm.size();
	  if (max_layers_ > 0) accept &= (layerCount <= max_layers_);
	  if (min_layers_ > 0) accept &= (layerCount >= min_layers_);


  }
  // return with final filter decision
  return accept;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPixelActivityFilter);
