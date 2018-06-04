#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/TrackReco/interface/Track.h"
//
// class declaration
//

class HLTPixelTrackFilter : public HLTFilter {
public:
  explicit HLTPixelTrackFilter(const edm::ParameterSet&);
  ~HLTPixelTrackFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
//  int countLayersWithClusters(edm::Handle<edmNew::DetSetVector<SiPixelCluster> > & clusterCol,const TrackerTopology& tTopo);

  edm::InputTag inputTag_;          // input tag identifying product containing pixel clusters
  unsigned int  min_pixelTracks_;      // minimum number of clusters
  unsigned int  max_pixelTracks_;      // maximum number of clusters
  edm::EDGetTokenT<reco::TrackCollection> inputToken_;

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

HLTPixelTrackFilter::HLTPixelTrackFilter(const edm::ParameterSet& config) : HLTFilter(config),
  inputTag_     (config.getParameter<edm::InputTag>("pixelTracks")),
  min_pixelTracks_ (config.getParameter<unsigned int>("maxPixelTracks")),
  max_pixelTracks_ (config.getParameter<unsigned int>("minPixelTracks"))
{
  inputToken_ = consumes< reco::TrackCollection >(inputTag_);
  LogDebug("") << "Using the " << inputTag_ << " input collection";
  LogDebug("") << "Requesting at least " << min_pixelTracks_ << " PixelTracks";
  if(max_pixelTracks_ > 0)
    LogDebug("") << "...but no more than " << max_pixelTracks_ << " PixelTracks";
}

HLTPixelTrackFilter::~HLTPixelTrackFilter() = default;

void
HLTPixelTrackFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("pixelTracks",edm::InputTag("hltPixelTracks"));
  desc.add<unsigned int>("minPixelTracks",0);
  desc.add<unsigned int>("maxPixelTracks",0);
  descriptions.add("hltPixelTrackFilter",desc);

}

//
// member functions
//
// ------------ method called to produce the data  ------------
bool HLTPixelTrackFilter::hltFilter(edm::Event& event, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);

  // get hold of products from Event
  edm::Handle< reco::TrackCollection> trackColl;
  event.getByToken(inputToken_, trackColl);

  unsigned int numTracks = trackColl->size();
  LogDebug("") << "Number of tracks accepted: " << numTracks;
  bool accept = (numTracks >= min_pixelTracks_);
  if(max_pixelTracks_ > 0)
    accept &= (numTracks <= max_pixelTracks_);
  std::cout << numTracks << std::endl;
  // return with final filter decision
  return accept;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPixelTrackFilter);
