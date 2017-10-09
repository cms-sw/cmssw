#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

//
// class decleration
//

class SeedClusterRemoverPhase2 : public edm::stream::EDProducer<> {

  public:
    SeedClusterRemoverPhase2(const edm::ParameterSet& iConfig) ;
    ~SeedClusterRemoverPhase2() override ;
    void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override ;
  private:
    bool doOuterTracker_, doPixel_;
    bool mergeOld_;
    typedef edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > PixelMaskContainer;
    typedef edm::ContainerMask<edmNew::DetSetVector<Phase2TrackerCluster1D> > Phase2OTMaskContainer;
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
    edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D> > phase2OTClusters_;
    edm::EDGetTokenT<reco::ClusterRemovalInfo> oldRemovalInfo_;
    edm::EDGetTokenT<PixelMaskContainer> oldPxlMaskToken_;
    edm::EDGetTokenT<Phase2OTMaskContainer> oldPh2OTMaskToken_;
    edm::EDGetTokenT<TrajectorySeedCollection> trajectories_;

    std::vector<uint8_t> pixels, OTs;                // avoid unneed alloc/dealloc of this
    edm::ProductID pixelSourceProdID, outerTrackerSourceProdID; // ProdIDs refs must point to (for consistency tests)

    inline void process(const TrackingRecHit *hit, float chi2, const TrackerGeometry* tg);

    template<typename T> 
    std::unique_ptr<edmNew::DetSetVector<T> >
    cleanup(const edmNew::DetSetVector<T> &oldClusters, const std::vector<uint8_t> &isGood, 
                reco::ClusterRemovalInfo::Indices &refs, const reco::ClusterRemovalInfo::Indices *oldRefs) ;

    // Carries in full removal info about a given det from oldRefs
    void mergeOld(reco::ClusterRemovalInfo::Indices &refs, const reco::ClusterRemovalInfo::Indices &oldRefs) ;

    bool clusterWasteSolution_;
    bool filterTracks_;
    reco::TrackBase::TrackQuality trackQuality_;
    std::vector<bool> collectedOuterTrackers_;
    std::vector<bool> collectedPixels_;
};


using namespace std;
using namespace edm;
using namespace reco;

SeedClusterRemoverPhase2::SeedClusterRemoverPhase2(const ParameterSet& iConfig):
    doOuterTracker_(iConfig.existsAs<bool>("doOuterTracker") ? iConfig.getParameter<bool>("doOuterTracker") : true),
    doPixel_(iConfig.existsAs<bool>("doPixel") ? iConfig.getParameter<bool>("doPixel") : true),
    mergeOld_(iConfig.exists("oldClusterRemovalInfo")),
    clusterWasteSolution_(true)
{
  if (iConfig.exists("clusterLessSolution"))
    clusterWasteSolution_=!iConfig.getParameter<bool>("clusterLessSolution");
  if (doPixel_ && clusterWasteSolution_) produces< edmNew::DetSetVector<SiPixelCluster> >();
  if (doOuterTracker_ && clusterWasteSolution_) produces< edmNew::DetSetVector<Phase2TrackerCluster1D> >();
  if (clusterWasteSolution_) produces< ClusterRemovalInfo >();

  if (!clusterWasteSolution_){
    produces<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > >();
    produces<edm::ContainerMask<edmNew::DetSetVector<Phase2TrackerCluster1D> > >();
  }
  trackQuality_=reco::TrackBase::undefQuality;
  filterTracks_=false;
  if (iConfig.exists("TrackQuality")){
    filterTracks_=true;
    trackQuality_=reco::TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));
  }

  trajectories_ = consumes<TrajectorySeedCollection>(iConfig.getParameter<InputTag>("trajectories"));
  if (doPixel_) pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<InputTag>("pixelClusters"));
  if (doOuterTracker_) phase2OTClusters_ = consumes<edmNew::DetSetVector<Phase2TrackerCluster1D> >(iConfig.getParameter<InputTag>("phase2OTClusters"));
  if (mergeOld_) {
    oldRemovalInfo_ = consumes<ClusterRemovalInfo>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
    oldPxlMaskToken_ = consumes<PixelMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
    oldPh2OTMaskToken_ = consumes<Phase2OTMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
  }
}


SeedClusterRemoverPhase2::~SeedClusterRemoverPhase2()
{
}

void SeedClusterRemoverPhase2::mergeOld(ClusterRemovalInfo::Indices &refs,
                                            const ClusterRemovalInfo::Indices &oldRefs) 
{
        for (size_t i = 0, n = refs.size(); i < n; ++i) {
            refs[i] = oldRefs[refs[i]];
       }
}

 
template<typename T> 
std::unique_ptr<edmNew::DetSetVector<T> >
SeedClusterRemoverPhase2::cleanup(const edmNew::DetSetVector<T> &oldClusters, const std::vector<uint8_t> &isGood, 
			     reco::ClusterRemovalInfo::Indices &refs, const reco::ClusterRemovalInfo::Indices *oldRefs){
    typedef typename edmNew::DetSetVector<T>             DSV;
    typedef typename edmNew::DetSetVector<T>::FastFiller DSF;
    typedef typename edmNew::DetSet<T>                   DS;
    auto output = std::make_unique<DSV>();
    output->reserve(oldClusters.size(), oldClusters.dataSize());

    unsigned int countOld=0;
    unsigned int countNew=0;

    // cluster removal loop
    const T * firstOffset = & oldClusters.data().front();
    for (typename DSV::const_iterator itdet = oldClusters.begin(), enddet = oldClusters.end(); itdet != enddet; ++itdet) {
        DS oldDS = *itdet;

        if (oldDS.empty()) continue; // skip empty detsets

        uint32_t id = oldDS.detId();
        DSF outds(*output, id);

        for (typename DS::const_iterator it = oldDS.begin(), ed = oldDS.end(); it != ed; ++it) {
            uint32_t index = ((&*it) - firstOffset);
	    countOld++;
            if (isGood[index]) { 
                outds.push_back(*it);
		countNew++;
                refs.push_back(index); 
                LogDebug("SeedClusterRemoverPhase2") << "SeedClusterRemoverPhase2::cleanup " << typeid(T).name() << " reference " << index << " to " << (refs.size() - 1);
            }

	}
        if (outds.empty()) outds.abort(); // not write in an empty DSV
    }

    if (oldRefs != nullptr) mergeOld(refs, *oldRefs);
    return output;
}




void SeedClusterRemoverPhase2::process(const TrackingRecHit *hit, float chi2, const TrackerGeometry* tg) {

  DetId detid = hit->geographicalId(); 
  uint32_t subdet = detid.subdetId();

  assert(subdet > 0);

  const type_info &hitType = typeid(*hit);
  if (hitType == typeid(SiPixelRecHit)) {

    if(!doPixel_) return;

    const SiPixelRecHit *pixelHit = static_cast<const SiPixelRecHit *>(hit);
    SiPixelRecHit::ClusterRef cluster = pixelHit->cluster();
    LogDebug("SeedClusterRemoverPhase2") << "Plain Pixel RecHit in det " << detid.rawId();

    if (cluster.id() != pixelSourceProdID) throw cms::Exception("Inconsistent Data") << 
            "SeedClusterRemoverPhase2: pixel cluster ref from Product ID = " << cluster.id() << 
            " does not match with source cluster collection (ID = " << pixelSourceProdID << ")\n.";

    assert(cluster.id() == pixelSourceProdID);

    // mark as used
    pixels[cluster.key()] = false;
    
    assert(collectedPixels_.size() > cluster.key());
    if(!clusterWasteSolution_) collectedPixels_[cluster.key()]=true;
      
  } else if (hitType == typeid(Phase2TrackerRecHit1D)) {

    if(!doOuterTracker_) return;

    const Phase2TrackerRecHit1D *ph2OThit = static_cast<const Phase2TrackerRecHit1D*>(hit);
    LogDebug("SeedClusterRemoverPhase2") << "Plain Phase2TrackerRecHit1D in det " << detid.rawId();

    Phase2TrackerRecHit1D::CluRef cluster = ph2OThit->cluster();
    if (cluster.id() != outerTrackerSourceProdID) throw cms::Exception("Inconsistent Data") <<
      "SeedClusterRemoverPhase2: strip cluster ref from Product ID = " << cluster.id() <<
      " does not match with source cluster collection (ID = " << outerTrackerSourceProdID << ")\n.";
  
    assert(cluster.id() == outerTrackerSourceProdID);

    OTs[cluster.key()] = false;
    assert(collectedOuterTrackers_.size() > cluster.key());
    if (!clusterWasteSolution_) collectedOuterTrackers_[cluster.key()]=true;
  
  } else throw cms::Exception("NOT IMPLEMENTED") << "I received a hit that was neither SiPixelRecHit nor Phase2TrackerRecHit1D but " << hitType.name() << " on detid " << detid.rawId() << "\n";

  
}

void
SeedClusterRemoverPhase2::produce(Event& iEvent, const EventSetup& iSetup)
{
    ProductID pixelOldProdID, stripOldProdID;

    edm::ESHandle<TrackerGeometry> tgh;
    iSetup.get<TrackerDigiGeometryRecord>().get("",tgh);  //is it correct to use "" ?

    Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
    if (doPixel_) {
        iEvent.getByToken(pixelClusters_, pixelClusters);
        pixelSourceProdID = pixelClusters.id();
    }
    LogDebug("SeedClusterRemoverPhase2") << "Read pixel with id " << pixelSourceProdID << std::endl;

    Handle<edmNew::DetSetVector<Phase2TrackerCluster1D> > phase2OTClusters;
    if (doOuterTracker_) {
        iEvent.getByToken(phase2OTClusters_, phase2OTClusters);
        outerTrackerSourceProdID = phase2OTClusters.id();
    }
    LogDebug("SeedClusterRemoverPhase2") << "Read OT cluster with id " << outerTrackerSourceProdID << std::endl;

    std::unique_ptr<ClusterRemovalInfo> cri;
    if (clusterWasteSolution_){
      if (doOuterTracker_ && doPixel_) cri = std::make_unique<ClusterRemovalInfo>(pixelClusters, phase2OTClusters);
      else if (doOuterTracker_) cri = std::make_unique<ClusterRemovalInfo>(phase2OTClusters);
      else if (doPixel_) cri = std::make_unique<ClusterRemovalInfo>(pixelClusters);
    }

    Handle<ClusterRemovalInfo> oldRemovalInfo;
    if (mergeOld_ && clusterWasteSolution_) {
        iEvent.getByToken(oldRemovalInfo_, oldRemovalInfo); 
        // Check ProductIDs
        if ( (oldRemovalInfo->stripNewRefProd().id() == phase2OTClusters.id()) &&
             (oldRemovalInfo->pixelNewRefProd().id() == pixelClusters.id()) ) {

            cri->getOldClustersFrom(*oldRemovalInfo);

            pixelOldProdID = oldRemovalInfo->pixelRefProd().id();
            stripOldProdID = oldRemovalInfo->stripRefProd().id();

        } else {
	    edm::EDConsumerBase::Labels labels;
	    labelsForToken(oldRemovalInfo_,labels);
            throw cms::Exception("Inconsistent Data") << "SeedClusterRemoverPhase2: " <<
                "Input collection product IDs are [pixel: " << pixelClusters.id() << ", strip: " << phase2OTClusters.id() << "] \n" <<
                "\t but the *old* ClusterRemovalInfo " << labels.productInstance << " refers as 'new product ids' to " <<
                    "[pixel: " << oldRemovalInfo->pixelNewRefProd().id() << ", strip: " << oldRemovalInfo->stripNewRefProd().id() << "]\n" << 
                "NOTA BENE: when running SeedClusterRemoverPhase2 with an old ClusterRemovalInfo the hits in the trajectory MUST be already re-keyed.\n";
        }
    } else { // then Old == Source
        pixelOldProdID = pixelSourceProdID;
        stripOldProdID = outerTrackerSourceProdID;
    }

    if (doOuterTracker_) {
      OTs.resize(phase2OTClusters->dataSize()); fill(OTs.begin(), OTs.end(), true);
    }
    if (doPixel_) {
      pixels.resize(pixelClusters->dataSize()); fill(pixels.begin(), pixels.end(), true);
    }
    if(mergeOld_) {
      edm::Handle<PixelMaskContainer> oldPxlMask;
      edm::Handle<Phase2OTMaskContainer> oldPh2OTMask;
      iEvent.getByToken(oldPxlMaskToken_ ,oldPxlMask);
      iEvent.getByToken(oldPh2OTMaskToken_ ,oldPh2OTMask);
      LogDebug("SeedClusterRemoverPhase2") << "to merge in, "<<oldPh2OTMask->size()<<" strp and "<<oldPxlMask->size()<<" pxl";
      oldPh2OTMask->copyMaskTo(collectedOuterTrackers_);
      oldPxlMask->copyMaskTo(collectedPixels_);
      assert(phase2OTClusters->dataSize()>=collectedOuterTrackers_.size());
      collectedOuterTrackers_.resize(phase2OTClusters->dataSize(),false);  // for ondemand
    }else {
      collectedOuterTrackers_.resize(phase2OTClusters->dataSize(), false);
      collectedPixels_.resize(pixelClusters->dataSize(), false);
    } 


    edm::Handle<TrajectorySeedCollection> seeds;
    iEvent.getByToken(trajectories_,seeds);

    TrajectorySeedCollection::const_iterator seed=seeds->begin();

    for (;seed!=seeds->end();++seed){
      TrajectorySeed::range hits=seed->recHits();
      TrajectorySeed::const_iterator hit=hits.first;
      for (;hit!=hits.second;++hit){
	if (!hit->isValid()) continue;
	process( &(*hit), 0. , tgh.product());
      }
    }
      
    
    if (doPixel_ && clusterWasteSolution_) {
        OrphanHandle<edmNew::DetSetVector<SiPixelCluster> > newPixels =
          iEvent.put(cleanup(*pixelClusters, pixels, cri->pixelIndices(), mergeOld_ ? &oldRemovalInfo->pixelIndices() : nullptr));
        LogDebug("SeedClusterRemoverPhase2") << "SeedClusterRemoverPhase2: Wrote pixel " << newPixels.id() << " from " << pixelSourceProdID ;
        cri->setNewPixelClusters(newPixels);
    }
    if (doOuterTracker_ && clusterWasteSolution_) {
        OrphanHandle<edmNew::DetSetVector<Phase2TrackerCluster1D> > newOuterTrackers =
          iEvent.put(cleanup(*phase2OTClusters, OTs, cri->stripIndices(), mergeOld_ ? &oldRemovalInfo->stripIndices() : nullptr));
        LogDebug("SeedClusterRemoverPhase2") << "SeedClusterRemoverPhase2: Wrote strip " << newOuterTrackers.id() << " from " << outerTrackerSourceProdID;
        cri->setNewPhase2OTClusters(newOuterTrackers);
    }

    
    if (clusterWasteSolution_) {
      iEvent.put(std::move(cri));
    }

    pixels.clear(); OTs.clear(); 

    if (!clusterWasteSolution_){
      
      LogDebug("SeedClusterRemoverPhase2")<<"total strip to skip: "<<std::count(collectedOuterTrackers_.begin(),collectedOuterTrackers_.end(),true);
      iEvent.put(std::make_unique<Phase2OTMaskContainer>(edm::RefProd<edmNew::DetSetVector<Phase2TrackerCluster1D> >(phase2OTClusters),collectedOuterTrackers_));

      LogDebug("SeedClusterRemoverPhase2")<<"total pxl to skip: "<<std::count(collectedPixels_.begin(),collectedPixels_.end(),true);
      iEvent.put(std::make_unique<PixelMaskContainer>(edm::RefProd<edmNew::DetSetVector<SiPixelCluster> >(pixelClusters),collectedPixels_));
      
    }
    collectedOuterTrackers_.clear();
    collectedPixels_.clear();    

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SeedClusterRemoverPhase2);
