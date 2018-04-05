#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"


#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"

#include "TrackingTools/PatternTools/interface/TrackCollectionTokens.h"

#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"

#include<limits>

namespace {

  class TrackClusterRemover final : public edm::global::EDProducer<> {
  public:
    TrackClusterRemover(const edm::ParameterSet& iConfig) ;
    ~TrackClusterRemover() override{}
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  private:
    
    void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const override;
 
    using PixelMaskContainer    = edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster>>;
    using StripMaskContainer    = edm::ContainerMask<edmNew::DetSetVector<SiStripCluster>>;

    using QualityMaskCollection = std::vector<unsigned char>;

    const unsigned char maxChi2x5_;
    const int minNumberOfLayersWithMeasBeforeFiltering_;
    const reco::TrackBase::TrackQuality trackQuality_;


    const TrackCollectionTokens trajectories_;
    edm::EDGetTokenT<QualityMaskCollection> srcQuals;
    
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClusters_;
    
    edm::EDGetTokenT<PixelMaskContainer> oldPxlMaskToken_;
    edm::EDGetTokenT<StripMaskContainer> oldStrMaskToken_;
    
    // backward compatibility during transition period
    edm::EDGetTokenT<edm::ValueMap<int>>  overrideTrkQuals_;
    
  };

  void TrackClusterRemover::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("trajectories",edm::InputTag());
    desc.add<edm::InputTag>("trackClassifier",edm::InputTag("","QualityMasks"));
    desc.add<edm::InputTag>("pixelClusters",edm::InputTag("siPixelClusters"));
    desc.add<edm::InputTag>("stripClusters",edm::InputTag("siStripClusters"));
    desc.add<edm::InputTag>("oldClusterRemovalInfo",edm::InputTag());

    desc.add<std::string>("TrackQuality","highPurity");
    desc.add<double>("maxChi2",30.);
    desc.add<int>("minNumberOfLayersWithMeasBeforeFiltering",0);
    // old mode     
    desc.add<edm::InputTag>("overrideTrkQuals",edm::InputTag());

    descriptions.add("trackClusterRemover", desc);
 
  }
  
  TrackClusterRemover::TrackClusterRemover(const edm::ParameterSet& iConfig) :
    maxChi2x5_(Traj2TrackHits::toChi2x5(iConfig.getParameter<double>("maxChi2"))),
    minNumberOfLayersWithMeasBeforeFiltering_(iConfig.getParameter<int>("minNumberOfLayersWithMeasBeforeFiltering")),
    trackQuality_(reco::TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"))),

    trajectories_(iConfig.getParameter<edm::InputTag>("trajectories"),consumesCollector())

  {


    auto const &  pixelClusters = iConfig.getParameter<edm::InputTag>("pixelClusters");
    auto const &  stripClusters = iConfig.getParameter<edm::InputTag>("stripClusters");

    if (pixelClusters.label().empty() && stripClusters.label().empty() )  throw edm::Exception(edm::errors::Configuration) << "Configuration Error: TrackClusterRemover used without input cluster collections";
    if ( !pixelClusters.label().empty() ){
		 pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(pixelClusters);
    		 produces<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > >();
    }
    if ( !stripClusters.label().empty() ){
		 stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(stripClusters);
    		 produces<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > >();
    }
    // old mode
    auto const & overrideTrkQuals = iConfig.getParameter<edm::InputTag>("overrideTrkQuals");
    if ( !overrideTrkQuals.label().empty() )
      overrideTrkQuals_ = consumes<edm::ValueMap<int> >(overrideTrkQuals);

    auto const & classifier = iConfig.getParameter<edm::InputTag>("trackClassifier");
    if ( !classifier.label().empty())
      srcQuals = consumes<QualityMaskCollection>(classifier);

    auto const &  oldClusterRemovalInfo = iConfig.getParameter<edm::InputTag>("oldClusterRemovalInfo");
    if (!oldClusterRemovalInfo.label().empty()) {
      oldPxlMaskToken_ = consumes<PixelMaskContainer>(oldClusterRemovalInfo);
      oldStrMaskToken_ = consumes<StripMaskContainer>(oldClusterRemovalInfo);
    }

  }


  void
  TrackClusterRemover::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const
  {

 
    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
    if(!pixelClusters_.isUninitialized()) iEvent.getByToken(pixelClusters_, pixelClusters);
    edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
    if(!stripClusters_.isUninitialized()) iEvent.getByToken(stripClusters_, stripClusters);


    std::vector<bool> collectedStrips;
    std::vector<bool> collectedPixels;

    if(!oldPxlMaskToken_.isUninitialized()) {
      edm::Handle<PixelMaskContainer> oldPxlMask;
      edm::Handle<StripMaskContainer> oldStrMask;
      iEvent.getByToken(oldPxlMaskToken_ ,oldPxlMask);
      iEvent.getByToken(oldStrMaskToken_ ,oldStrMask);
      LogDebug("TrackClusterRemover")<<"to merge in, "<<oldStrMask->size()<<" strp, "<<oldPxlMask->size()<<" pxl";
   
   // std::cout <<"TrackClusterRemover "<<"to merge in, "<<oldStrMask->size()<<" strp and "<<oldPxlMask->size()<<" pxl" << std::endl;
      oldStrMask->copyMaskTo(collectedStrips);
      oldPxlMask->copyMaskTo(collectedPixels);
      if(!stripClusters_.isUninitialized()){
      	assert(stripClusters->dataSize()>=collectedStrips.size());
      	collectedStrips.resize(stripClusters->dataSize(), false);
      //std::cout << "TrackClusterRemover " <<"total strip already to skip: "
      //	<<std::count(collectedStrips.begin(),collectedStrips.end(),true) <<std::endl;
      }
    }else {
      if(!stripClusters_.isUninitialized())  collectedStrips.resize(stripClusters->dataSize(), false);
      if(!pixelClusters_.isUninitialized())  collectedPixels.resize(pixelClusters->dataSize(), false);
    } 




    // loop over trajectories, filter, mask clusters../

   unsigned char qualMask = ~0;
   if (trackQuality_!=reco::TrackBase::undefQuality) qualMask = 1<<trackQuality_; 
 
    
    auto const & tracks = trajectories_.tracks(iEvent);
    auto s = tracks.size();

    // assert(s==trajs.size());

    QualityMaskCollection oldStyle;
    QualityMaskCollection const * pquals=nullptr;
    
    if (!overrideTrkQuals_.isUninitialized()) {
      edm::Handle<edm::ValueMap<int> > quals;
      iEvent.getByToken(overrideTrkQuals_,quals);
      assert(s==(*quals).size());

      oldStyle.resize(s,0);
      for (auto i=0U; i<s; ++i) if ( (*quals).get(i) > 0 ) oldStyle[i] = (255)&(*quals).get(i);
      pquals = &oldStyle; 
    }

    if (!srcQuals.isUninitialized()) {
      edm::Handle<QualityMaskCollection> hqual;
      iEvent.getByToken(srcQuals, hqual);
      pquals = hqual.product();
    }
    // if (!pquals) std::cout << "no qual collection" << std::endl;
    for (auto i=0U; i<s; ++i){
      const reco::Track & track = tracks[i];
      bool goodTk =  (pquals) ? (*pquals)[i] & qualMask : track.quality(trackQuality_);
      if ( !goodTk) continue;
      if(track.hitPattern().trackerLayersWithMeasurement() < minNumberOfLayersWithMeasBeforeFiltering_) continue;

      auto const & chi2sX5 = track.extra()->chi2sX5();
      assert(chi2sX5.size()==track.recHitsSize());
      auto hb = track.recHitsBegin();
      for(unsigned int h=0;h<track.recHitsSize();h++){
        auto recHit = *(hb+h);
	auto const & hit = *recHit;
	if (!hit.isValid()) continue;
	if ( chi2sX5[h] > maxChi2x5_ ) continue; // skip outliers
        auto const & thit = reinterpret_cast<BaseTrackerRecHit const&>(hit);
        auto const & cluster = thit.firstClusterRef();
	if(!stripClusters_.isUninitialized() && cluster.isStrip()) collectedStrips[cluster.key()]=true;
	if(!pixelClusters_.isUninitialized() && cluster.isPixel()) collectedPixels[cluster.key()]=true;
	if (trackerHitRTTI::isMatched(thit))
	  collectedStrips[reinterpret_cast<SiStripMatchedRecHit2D const&>(hit).stereoClusterRef().key()]=true;
      }
    }


    // std::cout << " => collectedStrips: " << collectedStrips.size() << std::endl;
    if(!stripClusters_.isUninitialized()){ 
    	auto removedStripClusterMask =
      		std::make_unique<StripMaskContainer>(edm::RefProd<edmNew::DetSetVector<SiStripCluster>>(stripClusters),collectedStrips);
      LogDebug("TrackClusterRemover")<<"total strip to skip: "<<std::count(collectedStrips.begin(),collectedStrips.end(),true);
      // std::cout << "TrackClusterRemover " <<"total strip to skip: "<<std::count(collectedStrips.begin(),collectedStrips.end(),true) <<std::endl;
      iEvent.put(std::move(removedStripClusterMask));
    }
    if(!pixelClusters_.isUninitialized()){
      auto removedPixelClusterMask= 
	std::make_unique<PixelMaskContainer>(edm::RefProd<edmNew::DetSetVector<SiPixelCluster>>(pixelClusters),collectedPixels);      
      LogDebug("TrackClusterRemover")<<"total pxl to skip: "<<std::count(collectedPixels.begin(),collectedPixels.end(),true);
      iEvent.put(std::move(removedPixelClusterMask));
    }

  }


}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackClusterRemover);

