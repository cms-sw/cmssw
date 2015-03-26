#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


namespace {

  class TrackClusterRemover : public edm::stream::EDProducer<> {
  public:
    TrackClusterRemover(const edm::ParameterSet& iConfig) ;
    ~TrackClusterRemover(){}
    void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override ;
  private:
 


    using PixelMaskContainer = edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster>>;
    using StripMaskContainer = edm::ContainerMask<edmNew::DetSetVector<SiStripCluster>>;


    bool mergeOld_;
   
    bool filterTracks_ = false;
    int minNumberOfLayersWithMeasBeforeFiltering_=0;
    float maxChi2_;
    reco::TrackBase::TrackQuality trackQuality_;


    edm::EDGetTokenT<TrajTrackAssociationCollection> trajectories_;

    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClusters_;

    edm::EDGetTokenT<PixelMaskContainer> oldPxlMaskToken_;
    edm::EDGetTokenT<StripMaskContainer> oldStrMaskToken_;
    std::vector< edm::EDGetTokenT<edm::ValueMap<int> > > overrideTrkQuals_;



  };


  TrackClusterRemover::TrackClusterRemover(const edm::ParameterSet& iConfig) :
    maxChi2_(iConfig.getParameter<double>("maxChi2"))
  {

    produces<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > >();
    produces<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > >();


    trajectories_  = consumes<TrajTrackAssociationCollection>       (iConfig.getParameter<edm::InputTag>("trajectories") );
    pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"));
    stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("stripClusters"));


    if (iConfig.exists("overrideTrkQuals")) {
      edm::InputTag overrideTrkQuals = iConfig.getParameter<edm::InputTag>("overrideTrkQuals");
      if ( !(overrideTrkQuals==edm::InputTag("")) ) 
	overrideTrkQuals_.push_back( consumes<edm::ValueMap<int> >(overrideTrkQuals) );
    }

    trackQuality_=reco::TrackBase::undefQuality;
    filterTracks_=false;
    if (iConfig.exists("TrackQuality")){
      filterTracks_=true;
      std::string trackQuality = iConfig.getParameter<std::string>("TrackQuality");
      if ( !trackQuality.empty() ) {
	trackQuality_=reco::TrackBase::qualityByName( trackQuality );
	minNumberOfLayersWithMeasBeforeFiltering_ = iConfig.existsAs<int>("minNumberOfLayersWithMeasBeforeFiltering") ? 
	  iConfig.getParameter<int>("minNumberOfLayersWithMeasBeforeFiltering") : 0;
      }
    }

    if ( iConfig.exists("oldClusterRemovalInfo") ) {
      edm::InputTag oldClusterRemovalInfo = iConfig.getParameter<edm::InputTag>("oldClusterRemovalInfo");
      mergeOld_ = ( (oldClusterRemovalInfo==edm::InputTag("")) ? false : true );
    } else
      mergeOld_ = false;

    if (mergeOld_) {
      oldPxlMaskToken_ = consumes<PixelMaskContainer>(iConfig.getParameter<edm::InputTag>("oldClusterRemovalInfo"));
      oldStrMaskToken_ = consumes<StripMaskContainer>(iConfig.getParameter<edm::InputTag>("oldClusterRemovalInfo"));
    }

  }


  void
  TrackClusterRemover::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {

 
    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
    iEvent.getByToken(pixelClusters_, pixelClusters);
    edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
    iEvent.getByToken(stripClusters_, stripClusters);


    std::vector<bool> collectedStrips;
    std::vector<bool> collectedPixels;

    if(mergeOld_) {
      edm::Handle<PixelMaskContainer> oldPxlMask;
      edm::Handle<StripMaskContainer> oldStrMask;
      iEvent.getByToken(oldPxlMaskToken_ ,oldPxlMask);
      iEvent.getByToken(oldStrMaskToken_ ,oldStrMask);
      LogDebug("TrackClusterRemover")<<"to merge in, "<<oldStrMask->size()<<" strp and "<<oldPxlMask->size()<<" pxl";
      oldStrMask->copyMaskTo(collectedStrips);
      oldPxlMask->copyMaskTo(collectedPixels);
      assert(stripClusters->dataSize()>=collectedStrips.size());
      collectedStrips.resize(stripClusters->dataSize(), false);
    }else {
      collectedStrips.resize(stripClusters->dataSize(), false);
      collectedPixels.resize(pixelClusters->dataSize(), false);
    } 




    // loop over trajectories, filter, mask clusters../

    edm::Handle<TrajTrackAssociationCollection> trajectories_totrack; 
    iEvent.getByToken(trajectories_,trajectories_totrack);

    std::vector<edm::Handle<edm::ValueMap<int> > > quals;
    
    if ( overrideTrkQuals_.size() > 0) {
      quals.resize(1);
      iEvent.getByToken(overrideTrkQuals_[0],quals[0]);
    }
   
    for (auto const &  asst : *trajectories_totrack){
      const reco::Track & track = *(asst.val);
      if (filterTracks_) {
	bool goodTk = true;
	if ( quals.size()!=0) {
	  int qual=(*(quals[0]))[asst.val];
	  if ( qual < 0 ) {goodTk=false;}
	  //note that this does not work for some trackquals (goodIterative  or undefQuality)
	  else
	    goodTk = ( qual & (1<<trackQuality_))>>trackQuality_;
	}
	else
	  goodTk=(track.quality(trackQuality_));
	  if ( !goodTk) continue;
	  if(track.hitPattern().trackerLayersWithMeasurement() < minNumberOfLayersWithMeasBeforeFiltering_) continue;
      }
      const Trajectory &tj = *(asst.key);
      const auto & tms = tj.measurements();
      for (auto const & tm :  tms) {
	auto const & hit = *tm.recHit();
	if (!hit.isValid()) continue; 
	if ( tm.estimate() > maxChi2_ ) continue; // skip outliers
        auto const & thit = reinterpret_cast<BaseTrackerRecHit const&>(hit);
        auto const & cluster = thit.firstClusterRef();
	if (cluster.isStrip()) collectedStrips[cluster.key()]=true;
	else                  collectedPixels[cluster.key()]=true;
	if (trackerHitRTTI::isMatched(thit))
	  collectedStrips[reinterpret_cast<SiStripMatchedRecHit2D const&>(hit).stereoClusterRef().key()]=true;
      }
    }


    std::auto_ptr<StripMaskContainer> removedStripClusterMask(
         new StripMaskContainer(edm::RefProd<edmNew::DetSetVector<SiStripCluster> >(stripClusters),collectedStrips));
      LogDebug("TrackClusterRemover")<<"total strip to skip: "<<std::count(collectedStrips.begin(),collectedStrips.end(),true);
      // std::cout << "TrackClusterRemover " <<"total strip to skip: "<<std::count(collectedStrips_.begin(),collectedStrips_.end(),true) <<std::endl;
       iEvent.put( removedStripClusterMask );

      std::auto_ptr<PixelMaskContainer> removedPixelClusterMask(
         new PixelMaskContainer(edm::RefProd<edmNew::DetSetVector<SiPixelCluster> >(pixelClusters),collectedPixels));      
      LogDebug("TrackClusterRemover")<<"total pxl to skip: "<<std::count(collectedPixels.begin(),collectedPixels.end(),true);
      iEvent.put( removedPixelClusterMask );
 


  }


}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackClusterRemover);

