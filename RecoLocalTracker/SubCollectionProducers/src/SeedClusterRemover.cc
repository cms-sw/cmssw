#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
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


//
// class decleration
//

class SeedClusterRemover : public edm::stream::EDProducer<> {
    public:
        SeedClusterRemover(const edm::ParameterSet& iConfig) ;
        ~SeedClusterRemover() ;
        void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override ;
    private:
        struct ParamBlock {
            ParamBlock() : isSet_(false), usesCharge_(false) {}
            ParamBlock(const edm::ParameterSet& iConfig) :
                isSet_(true), 
                usesCharge_(iConfig.exists("maxCharge")),
                usesSize_(iConfig.exists("maxSize")),
                maxChi2_(iConfig.getParameter<double>("maxChi2")),
                maxCharge_(usesCharge_ ? iConfig.getParameter<double>("maxCharge") : 0), 
                maxSize_(usesSize_ ? iConfig.getParameter<uint32_t>("maxSize") : 0) { }
            bool  isSet_, usesCharge_, usesSize_;
            float maxChi2_, maxCharge_;
            size_t maxSize_;
        };
        static const unsigned int NumberOfParamBlocks = 6;

        bool doStrip_, doPixel_;
        bool mergeOld_;
	typedef edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > PixelMaskContainer;
	typedef edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > StripMaskContainer;
	edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
	edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClusters_;
	edm::EDGetTokenT<reco::ClusterRemovalInfo> oldRemovalInfo_;
	edm::EDGetTokenT<PixelMaskContainer> oldPxlMaskToken_;
	edm::EDGetTokenT<StripMaskContainer> oldStrMaskToken_;
	edm::EDGetTokenT<TrajectorySeedCollection> trajectories_;

//         std::vector< edm::EDGetTokenT<edm::ValueMap<int> > > overrideTrkQuals_;

        ParamBlock pblocks_[NumberOfParamBlocks];
        void readPSet(const edm::ParameterSet& iConfig, const std::string &name, 
                int id1=-1, int id2=-1, int id3=-1, int id4=-1, int id5=-1, int id6=-1) ;

        std::vector<uint8_t> pixels, strips;                // avoid unneed alloc/dealloc of this
        edm::ProductID pixelSourceProdID, stripSourceProdID; // ProdIDs refs must point to (for consistency tests)

        inline void process(const TrackingRecHit *hit, float chi2);
        inline void process(const OmniClusterRef & cluRef, uint32_t subdet);


        template<typename T> 
        std::auto_ptr<edmNew::DetSetVector<T> >
        cleanup(const edmNew::DetSetVector<T> &oldClusters, const std::vector<uint8_t> &isGood, 
                    reco::ClusterRemovalInfo::Indices &refs, const reco::ClusterRemovalInfo::Indices *oldRefs) ;

        // Carries in full removal info about a given det from oldRefs
        void mergeOld(reco::ClusterRemovalInfo::Indices &refs, const reco::ClusterRemovalInfo::Indices &oldRefs) ;

  bool clusterWasteSolution_;
  bool filterTracks_;
  reco::TrackBase::TrackQuality trackQuality_;
  std::vector<bool> collectedStrips_;
  std::vector<bool> collectedPixels_;
};


using namespace std;
using namespace edm;
using namespace reco;

void
SeedClusterRemover::readPSet(const edm::ParameterSet& iConfig, const std::string &name, 
    int id1, int id2, int id3, int id4, int id5, int id6) 
{
    if (iConfig.exists(name)) {
        ParamBlock pblock(iConfig.getParameter<ParameterSet>(name));
        if (id1 == -1) {
            fill(pblocks_, pblocks_+NumberOfParamBlocks, pblock);
        } else {
            pblocks_[id1] = pblock;
            if (id2 != -1) pblocks_[id2] = pblock;
            if (id3 != -1) pblocks_[id3] = pblock;
            if (id4 != -1) pblocks_[id4] = pblock;
            if (id5 != -1) pblocks_[id5] = pblock;
            if (id6 != -1) pblocks_[id6] = pblock;
        }
    }
}

SeedClusterRemover::SeedClusterRemover(const ParameterSet& iConfig):
    doStrip_(iConfig.existsAs<bool>("doStrip") ? iConfig.getParameter<bool>("doStrip") : true),
    doPixel_(iConfig.existsAs<bool>("doPixel") ? iConfig.getParameter<bool>("doPixel") : true),
    mergeOld_(iConfig.exists("oldClusterRemovalInfo")),
    clusterWasteSolution_(true)
{
//   if (iConfig.exists("overrideTrkQuals"))
//     overrideTrkQuals_.push_back(iConfig.getParameter<edm::InputTag>("overrideTrkQuals"));
  if (iConfig.exists("clusterLessSolution"))
    clusterWasteSolution_=!iConfig.getParameter<bool>("clusterLessSolution");
  if (doPixel_ && clusterWasteSolution_) produces< edmNew::DetSetVector<SiPixelCluster> >();
  if (doStrip_ && clusterWasteSolution_) produces< edmNew::DetSetVector<SiStripCluster> >();
  if (clusterWasteSolution_) produces< ClusterRemovalInfo >();

    fill(pblocks_, pblocks_+NumberOfParamBlocks, ParamBlock());
    readPSet(iConfig, "Common",-1);
    if (doPixel_) {
        readPSet(iConfig, "Pixel" ,0,1);
        readPSet(iConfig, "PXB" ,0);
        readPSet(iConfig, "PXE" ,1);
    }
    if (doStrip_) {
        readPSet(iConfig, "Strip" ,2,3,4,5);
        readPSet(iConfig, "StripInner" ,2,3);
        readPSet(iConfig, "StripOuter" ,4,5);
        readPSet(iConfig, "TIB" ,2);
        readPSet(iConfig, "TID" ,3);
        readPSet(iConfig, "TOB" ,4);
        readPSet(iConfig, "TEC" ,5);
    }

    bool usingCharge = false;
    for (size_t i = 0; i < NumberOfParamBlocks; ++i) {
        if (!pblocks_[i].isSet_) throw cms::Exception("Configuration Error") << "SeedClusterRemover: Missing configuration for detector with subDetID = " << (i+1);
        if (pblocks_[i].usesCharge_ && !usingCharge) {
            throw cms::Exception("Configuration Error") << "SeedClusterRemover: Configuration for subDetID = " << (i+1) << " uses cluster charge, which is not enabled.";
        }
    }

    if (!clusterWasteSolution_){
      produces<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > >();
      produces<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > >();
    }
    trackQuality_=reco::TrackBase::undefQuality;
    filterTracks_=false;
    if (iConfig.exists("TrackQuality")){
      filterTracks_=true;
      trackQuality_=reco::TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));
    }

    trajectories_ = consumes<TrajectorySeedCollection>(iConfig.getParameter<InputTag>("trajectories"));
    if (doPixel_) pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<InputTag>("pixelClusters"));
    if (doStrip_) stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<InputTag>("stripClusters"));
    if (mergeOld_) {
      oldRemovalInfo_ = consumes<ClusterRemovalInfo>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
      oldPxlMaskToken_ = consumes<PixelMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
      oldStrMaskToken_ = consumes<StripMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
    }
}


SeedClusterRemover::~SeedClusterRemover()
{
}

void SeedClusterRemover::mergeOld(ClusterRemovalInfo::Indices &refs,
                                            const ClusterRemovalInfo::Indices &oldRefs) 
{
        for (size_t i = 0, n = refs.size(); i < n; ++i) {
            refs[i] = oldRefs[refs[i]];
       }
}

 
template<typename T> 
auto_ptr<edmNew::DetSetVector<T> >
SeedClusterRemover::cleanup(const edmNew::DetSetVector<T> &oldClusters, const std::vector<uint8_t> &isGood, 
			     reco::ClusterRemovalInfo::Indices &refs, const reco::ClusterRemovalInfo::Indices *oldRefs){
    typedef typename edmNew::DetSetVector<T>             DSV;
    typedef typename edmNew::DetSetVector<T>::FastFiller DSF;
    typedef typename edmNew::DetSet<T>                   DS;
    auto_ptr<DSV> output(new DSV());
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
                //std::cout << "SeedClusterRemover::cleanup " << typeid(T).name() << " reference " << index << " to " << (refs.size() - 1) << std::endl;
            }

	}
        if (outds.empty()) outds.abort(); // not write in an empty DSV
    }

    if (oldRefs != 0) mergeOld(refs, *oldRefs);
    return output;
}


void SeedClusterRemover::process(OmniClusterRef const & ocluster, uint32_t subdet) {
    SiStripRecHit2D::ClusterRef cluster = ocluster.cluster_strip();
  if (cluster.id() != stripSourceProdID) throw cms::Exception("Inconsistent Data") <<
    "SeedClusterRemover: strip cluster ref from Product ID = " << cluster.id() <<
    " does not match with source cluster collection (ID = " << stripSourceProdID << ")\n.";
  
  assert(cluster.id() == stripSourceProdID);
  if (pblocks_[subdet-1].usesSize_ && (cluster->amplitudes().size() > pblocks_[subdet-1].maxSize_)) return;

  strips[cluster.key()] = false;  
  //if (!clusterWasteSolution_) collectedStrip[hit->geographicalId()].insert(cluster);
  assert(collectedStrips_.size() > cluster.key());
  //assert(hit->geographicalId() == cluster->geographicalId()); //This condition fails
  if (!clusterWasteSolution_) collectedStrips_[cluster.key()]=true;
}


void SeedClusterRemover::process(const TrackingRecHit *hit, float chi2) {
    DetId detid = hit->geographicalId(); 
    uint32_t subdet = detid.subdetId();

    assert ((subdet > 0) && (subdet <= NumberOfParamBlocks));

    // chi2 cut
    if (chi2 > pblocks_[subdet-1].maxChi2_) return;

    if ((subdet == PixelSubdetector::PixelBarrel) || (subdet == PixelSubdetector::PixelEndcap)) {
        if (!doPixel_) return;
        // this is a pixel, and i *know* it is
        const SiPixelRecHit *pixelHit = static_cast<const SiPixelRecHit *>(hit);

        SiPixelRecHit::ClusterRef cluster = pixelHit->cluster();

        if (cluster.id() != pixelSourceProdID) throw cms::Exception("Inconsistent Data") << 
                "SeedClusterRemover: pixel cluster ref from Product ID = " << cluster.id() << 
                " does not match with source cluster collection (ID = " << pixelSourceProdID << ")\n.";

        assert(cluster.id() == pixelSourceProdID);
//DBG// cout << "HIT NEW PIXEL DETID = " << detid.rawId() << ", Cluster [ " << cluster.key().first << " / " <<  cluster.key().second << " ] " << endl;

        // if requested, cut on cluster size
        if (pblocks_[subdet-1].usesSize_ && (cluster->pixels().size() > pblocks_[subdet-1].maxSize_)) return;

        // mark as used
        pixels[cluster.key()] = false;
	
	//if(!clusterWasteSolution_) collectedPixel[detid.rawId()].insert(cluster);
   assert(collectedPixels_.size() > cluster.key());
   //assert(detid.rawId() == cluster->geographicalId()); //This condition fails
	if(!clusterWasteSolution_) collectedPixels_[cluster.key()]=true;
	
    } else { // aka Strip
        if (!doStrip_) return;
        const type_info &hitType = typeid(*hit);
        if (hitType == typeid(SiStripRecHit2D)) {
            const SiStripRecHit2D *stripHit = static_cast<const SiStripRecHit2D *>(hit);
//DBG//     cout << "Plain RecHit 2D: " << endl;
            process(stripHit->omniClusterRef(),subdet);}
	else if (hitType == typeid(SiStripRecHit1D)) {
	  const SiStripRecHit1D *hit1D = static_cast<const SiStripRecHit1D *>(hit);
	  process(hit1D->omniClusterRef(),subdet);
        } else if (hitType == typeid(SiStripMatchedRecHit2D)) {
            const SiStripMatchedRecHit2D *matchHit = static_cast<const SiStripMatchedRecHit2D *>(hit);
//DBG//     cout << "Matched RecHit 2D: " << endl;
            process(matchHit->monoClusterRef(),subdet);
            process(matchHit->stereoClusterRef(),subdet);
        } else if (hitType == typeid(ProjectedSiStripRecHit2D)) {
            const ProjectedSiStripRecHit2D *projHit = static_cast<const ProjectedSiStripRecHit2D *>(hit);
//DBG//     cout << "Projected RecHit 2D: " << endl;
            process(projHit->originalHit().omniClusterRef(),subdet);
        } else throw cms::Exception("NOT IMPLEMENTED") << "Don't know how to handle " << hitType.name() << " on detid " << detid.rawId() << "\n";
    }
}

/*   Schematic picture of n-th step Iterative removal
 *   (that os removing clusters after n-th step tracking)
 *   clusters:    [ C1 ] -> [ C2 ] -> ... -> [ Cn ] -> [ Cn + 1 ]
 *                   ^                          ^           ^--- OUTPUT "new" ID
 *                   |-- before any removal     |----- Source clusters                                                                             
 *                   |-- OUTPUT "old" ID        |----- Hits in Traj. point here                       
 *                   |                          \----- Old ClusterRemovalInfo "new" ID                 
 *                   \-- Old ClusterRemovalInfo "old" ID                                                  
 */


void
SeedClusterRemover::produce(Event& iEvent, const EventSetup& iSetup)
{
    ProductID pixelOldProdID, stripOldProdID;

    Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
    if (doPixel_) {
        iEvent.getByToken(pixelClusters_, pixelClusters);
        pixelSourceProdID = pixelClusters.id();
    }
//DBG// std::cout << "SeedClusterRemover: Read pixel " << pixelClusters_.encode() << " = ID " << pixelSourceProdID << std::endl;

    Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
    if (doStrip_) {
        iEvent.getByToken(stripClusters_, stripClusters);
        stripSourceProdID = stripClusters.id();
    }
//DBG// std::cout << "SeedClusterRemover: Read strip " << stripClusters_.encode() << " = ID " << stripSourceProdID << std::endl;

    auto_ptr<ClusterRemovalInfo> cri;
    if (clusterWasteSolution_){
      if (doStrip_ && doPixel_) cri.reset(new ClusterRemovalInfo(pixelClusters, stripClusters));
      else if (doStrip_) cri.reset(new ClusterRemovalInfo(stripClusters));
      else if (doPixel_) cri.reset(new ClusterRemovalInfo(pixelClusters));
    }

    Handle<ClusterRemovalInfo> oldRemovalInfo;
    if (mergeOld_ && clusterWasteSolution_) {
        iEvent.getByToken(oldRemovalInfo_, oldRemovalInfo); 
        // Check ProductIDs
        if ( (oldRemovalInfo->stripNewRefProd().id() == stripClusters.id()) &&
             (oldRemovalInfo->pixelNewRefProd().id() == pixelClusters.id()) ) {

            cri->getOldClustersFrom(*oldRemovalInfo);

            pixelOldProdID = oldRemovalInfo->pixelRefProd().id();
            stripOldProdID = oldRemovalInfo->stripRefProd().id();

        } else {
	    edm::EDConsumerBase::Labels labels;
	    labelsForToken(oldRemovalInfo_,labels);
            throw cms::Exception("Inconsistent Data") << "SeedClusterRemover: " <<
                "Input collection product IDs are [pixel: " << pixelClusters.id() << ", strip: " << stripClusters.id() << "] \n" <<
                "\t but the *old* ClusterRemovalInfo " << labels.productInstance << " refers as 'new product ids' to " <<
                    "[pixel: " << oldRemovalInfo->pixelNewRefProd().id() << ", strip: " << oldRemovalInfo->stripNewRefProd().id() << "]\n" << 
                "NOTA BENE: when running SeedClusterRemover with an old ClusterRemovalInfo the hits in the trajectory MUST be already re-keyed.\n";
        }
    } else { // then Old == Source
        pixelOldProdID = pixelSourceProdID;
        stripOldProdID = stripSourceProdID;
    }

    if (doStrip_) {
      strips.resize(stripClusters->dataSize()); fill(strips.begin(), strips.end(), true);
    }
    if (doPixel_) {
      pixels.resize(pixelClusters->dataSize()); fill(pixels.begin(), pixels.end(), true);
    }
    if(mergeOld_) {
      edm::Handle<PixelMaskContainer> oldPxlMask;
      edm::Handle<StripMaskContainer> oldStrMask;
      iEvent.getByToken(oldPxlMaskToken_ ,oldPxlMask);
      iEvent.getByToken(oldStrMaskToken_ ,oldStrMask);
      LogDebug("SeedClusterRemover")<<"to merge in, "<<oldStrMask->size()<<" strp and "<<oldPxlMask->size()<<" pxl";
      oldStrMask->copyMaskTo(collectedStrips_);
      oldPxlMask->copyMaskTo(collectedPixels_);
    }else {
      collectedStrips_.resize(stripClusters->dataSize()); fill(collectedStrips_.begin(), collectedStrips_.end(), false);
      collectedPixels_.resize(pixelClusters->dataSize()); fill(collectedPixels_.begin(), collectedPixels_.end(), false);
    } 


    edm::Handle<TrajectorySeedCollection> seeds;
    iEvent.getByToken(trajectories_,seeds);

    TrajectorySeedCollection::const_iterator seed=seeds->begin();

    for (;seed!=seeds->end();++seed){
      TrajectorySeed::range hits=seed->recHits();
      TrajectorySeed::const_iterator hit=hits.first;
      for (;hit!=hits.second;++hit){
	if (!hit->isValid()) continue;
	process( &(*hit), 0. );
      }
    }
      
    
    if (doPixel_ && clusterWasteSolution_) {
        auto_ptr<edmNew::DetSetVector<SiPixelCluster> > newPixelClusters = cleanup(*pixelClusters, pixels, 
                    cri->pixelIndices(), mergeOld_ ? &oldRemovalInfo->pixelIndices() : 0);
        OrphanHandle<edmNew::DetSetVector<SiPixelCluster> > newPixels = iEvent.put(newPixelClusters); 
//DBG// std::cout << "SeedClusterRemover: Wrote pixel " << newPixels.id() << " from " << pixelSourceProdID << std::endl;
        cri->setNewPixelClusters(newPixels);
    }
    if (doStrip_ && clusterWasteSolution_) {
        auto_ptr<edmNew::DetSetVector<SiStripCluster> > newStripClusters = cleanup(*stripClusters, strips, 
                    cri->stripIndices(), mergeOld_ ? &oldRemovalInfo->stripIndices() : 0);
        OrphanHandle<edmNew::DetSetVector<SiStripCluster> > newStrips = iEvent.put(newStripClusters); 
//DBG// std::cout << "SeedClusterRemover: Wrote strip " << newStrips.id() << " from " << stripSourceProdID << std::endl;
        cri->setNewStripClusters(newStrips);
    }

    
    if (clusterWasteSolution_) {
      //      double fraction_pxl= cri->pixelIndices().size() / (double) pixels.size();
      //      double fraction_strp= cri->stripIndices().size() / (double) strips.size();
      //      edm::LogWarning("SeedClusterRemover")<<" fraction: " << fraction_pxl <<" "<<fraction_strp;
      iEvent.put(cri);
    }

    pixels.clear(); strips.clear(); 

    if (!clusterWasteSolution_){
      //auto_ptr<edmNew::DetSetVector<SiPixelClusterRefNew> > removedPixelClsuterRefs(new edmNew::DetSetVector<SiPixelClusterRefNew>());
      //auto_ptr<edmNew::DetSetVector<SiStripRecHit1D::ClusterRef> > removedStripClsuterRefs(new edmNew::DetSetVector<SiStripRecHit1D::ClusterRef>());
      
      std::auto_ptr<StripMaskContainer> removedStripClusterMask(
         new StripMaskContainer(edm::RefProd<edmNew::DetSetVector<SiStripCluster> >(stripClusters),collectedStrips_));
      LogDebug("SeedClusterRemover")<<"total strip to skip: "<<std::count(collectedStrips_.begin(),collectedStrips_.end(),true);
      iEvent.put( removedStripClusterMask );

      std::auto_ptr<PixelMaskContainer> removedPixelClusterMask(
         new PixelMaskContainer(edm::RefProd<edmNew::DetSetVector<SiPixelCluster> >(pixelClusters),collectedPixels_));      
      LogDebug("SeedClusterRemover")<<"total pxl to skip: "<<std::count(collectedPixels_.begin(),collectedPixels_.end(),true);
      iEvent.put( removedPixelClusterMask );
      
    }
    

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SeedClusterRemover);
