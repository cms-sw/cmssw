#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"


#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"


//
// class decleration
//

class HITrackClusterRemover : public edm::stream::EDProducer<> {
    public:
        HITrackClusterRemover(const edm::ParameterSet& iConfig) ;
        ~HITrackClusterRemover() ;
        void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override ;
    private:
        struct ParamBlock {
            ParamBlock() : isSet_(false), usesCharge_(false) {}
            ParamBlock(const edm::ParameterSet& iConfig) :
                isSet_(true), 
                usesCharge_(iConfig.exists("maxCharge")),
                usesSize_(iConfig.exists("maxSize")),
                cutOnPixelCharge_(iConfig.exists("minGoodPixelCharge")),
                cutOnStripCharge_(iConfig.exists("minGoodStripCharge")),
                maxChi2_(iConfig.getParameter<double>("maxChi2")),
                maxCharge_(usesCharge_ ? iConfig.getParameter<double>("maxCharge") : 0), 
                minGoodPixelCharge_(cutOnPixelCharge_ ? iConfig.getParameter<double>("minGoodPixelCharge") : 0), 
                minGoodStripCharge_(cutOnStripCharge_ ? iConfig.getParameter<double>("minGoodStripCharge") : 0), 
                maxSize_(usesSize_ ? iConfig.getParameter<uint32_t>("maxSize") : 0) { }
            bool  isSet_, usesCharge_, usesSize_, cutOnPixelCharge_, cutOnStripCharge_;
            float maxChi2_, maxCharge_, minGoodPixelCharge_, minGoodStripCharge_;
            size_t maxSize_;
        };
        static const unsigned int NumberOfParamBlocks = 6;

        bool doTracks_;
        bool doStrip_, doPixel_;
        bool mergeOld_;
	typedef edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > PixelMaskContainer;
	typedef edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > StripMaskContainer;
	edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
	edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClusters_;
	edm::EDGetTokenT<reco::TrackCollection> tracks_;
	edm::EDGetTokenT<reco::ClusterRemovalInfo> oldRemovalInfo_;
	edm::EDGetTokenT<PixelMaskContainer> oldPxlMaskToken_;
	edm::EDGetTokenT<StripMaskContainer> oldStrMaskToken_;
        std::vector< edm::EDGetTokenT<edm::ValueMap<int> > > overrideTrkQuals_;
	edm::EDGetTokenT<SiStripRecHit2DCollection> rphiRecHitToken_, stereoRecHitToken_;
// 	edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitsToken_;

        ParamBlock pblocks_[NumberOfParamBlocks];
        void readPSet(const edm::ParameterSet& iConfig, const std::string &name, 
                int id1=-1, int id2=-1, int id3=-1, int id4=-1, int id5=-1, int id6=-1) ;

        std::vector<uint8_t> pixels, strips;                // avoid unneed alloc/dealloc of this
        edm::ProductID pixelSourceProdID, stripSourceProdID; // ProdIDs refs must point to (for consistency tests)

        inline void process(const TrackingRecHit *hit, unsigned char chi2, const TrackerGeometry* tg);
        inline void process(const OmniClusterRef & cluRef, SiStripDetId & detid, bool fromTrack);


        template<typename T> 
        std::unique_ptr<edmNew::DetSetVector<T> >
        cleanup(const edmNew::DetSetVector<T> &oldClusters, const std::vector<uint8_t> &isGood, 
                    reco::ClusterRemovalInfo::Indices &refs, const reco::ClusterRemovalInfo::Indices *oldRefs) ;

        // Carries in full removal info about a given det from oldRefs
        void mergeOld(reco::ClusterRemovalInfo::Indices &refs, const reco::ClusterRemovalInfo::Indices &oldRefs) ;

  bool clusterWasteSolution_, doStripChargeCheck_, doPixelChargeCheck_;
  std::string stripRecHits_, pixelRecHits_;
  bool filterTracks_;
  int minNumberOfLayersWithMeasBeforeFiltering_;
  reco::TrackBase::TrackQuality trackQuality_;
  std::vector<bool> collectedStrips_;
  std::vector<bool> collectedPixels_;

  float sensorThickness (const SiStripDetId& detid) const;

};


using namespace std;
using namespace edm;
using namespace reco;

void
HITrackClusterRemover::readPSet(const edm::ParameterSet& iConfig, const std::string &name, 
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

HITrackClusterRemover::HITrackClusterRemover(const ParameterSet& iConfig):
    doTracks_(iConfig.exists("trajectories")),
    doStrip_(iConfig.existsAs<bool>("doStrip") ? iConfig.getParameter<bool>("doStrip") : true),
    doPixel_(iConfig.existsAs<bool>("doPixel") ? iConfig.getParameter<bool>("doPixel") : true),
    mergeOld_(iConfig.exists("oldClusterRemovalInfo")),
    clusterWasteSolution_(true),
    doStripChargeCheck_(iConfig.existsAs<bool>("doStripChargeCheck") ? iConfig.getParameter<bool>("doStripChargeCheck") : false),
    doPixelChargeCheck_(iConfig.existsAs<bool>("doPixelChargeCheck") ? iConfig.getParameter<bool>("doPixelChargeCheck") : false),
    stripRecHits_(doStripChargeCheck_ ? iConfig.getParameter<std::string>("stripRecHits") : std::string("siStripMatchedRecHits")),
    pixelRecHits_(doPixelChargeCheck_ ? iConfig.getParameter<std::string>("pixelRecHits") : std::string("siPixelRecHits"))
{
  mergeOld_ = mergeOld_ && iConfig.getParameter<InputTag>("oldClusterRemovalInfo").label()!="";
  if (iConfig.exists("overrideTrkQuals"))
    overrideTrkQuals_.push_back(consumes<edm::ValueMap<int> >(iConfig.getParameter<InputTag>("overrideTrkQuals")));
  if (iConfig.exists("clusterLessSolution"))
    clusterWasteSolution_=!iConfig.getParameter<bool>("clusterLessSolution");
  if ((doPixelChargeCheck_ && !doPixel_) || (doStripChargeCheck_ && !doStrip_))
      throw cms::Exception("Configuration Error") << "HITrackClusterRemover: Charge check asked without cluster collection ";
  if (doPixelChargeCheck_)
      throw cms::Exception("Configuration Error") << "HITrackClusterRemover: Pixel cluster charge check not yet implemented";

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
        if (!pblocks_[i].isSet_) throw cms::Exception("Configuration Error") << "HITrackClusterRemover: Missing configuration for detector with subDetID = " << (i+1);
        if (pblocks_[i].usesCharge_ && !usingCharge) {
            throw cms::Exception("Configuration Error") << "HITrackClusterRemover: Configuration for subDetID = " << (i+1) << " uses cluster charge, which is not enabled.";
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
      minNumberOfLayersWithMeasBeforeFiltering_ = iConfig.existsAs<int>("minNumberOfLayersWithMeasBeforeFiltering") ? 
	iConfig.getParameter<int>("minNumberOfLayersWithMeasBeforeFiltering") : 0;
    }

    if (doTracks_) tracks_ = consumes<reco::TrackCollection>(iConfig.getParameter<InputTag>("trajectories"));
    if (doPixel_) pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<InputTag>("pixelClusters"));
    if (doStrip_) stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<InputTag>("stripClusters"));
    if (mergeOld_) {
      oldRemovalInfo_ = consumes<ClusterRemovalInfo>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
      oldPxlMaskToken_ = consumes<PixelMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
      oldStrMaskToken_ = consumes<StripMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
    }

    if (doStripChargeCheck_) {
      rphiRecHitToken_ = consumes<SiStripRecHit2DCollection>(InputTag(stripRecHits_,"rphiRecHit"));
      stereoRecHitToken_ = consumes<SiStripRecHit2DCollection>(InputTag(stripRecHits_,"stereoRecHit"));
    }
//    if(doPixelChargeCheck_) pixelRecHitsToken_ = consumes<SiPixelRecHitCollection>(InputTag(pixelRecHits_));

}


HITrackClusterRemover::~HITrackClusterRemover()
{
}

void HITrackClusterRemover::mergeOld(ClusterRemovalInfo::Indices &refs,
                                            const ClusterRemovalInfo::Indices &oldRefs) 
{
        for (size_t i = 0, n = refs.size(); i < n; ++i) {
            refs[i] = oldRefs[refs[i]];
       }
}


template<typename T> 
std::unique_ptr<edmNew::DetSetVector<T> >
HITrackClusterRemover::cleanup(const edmNew::DetSetVector<T> &oldClusters, const std::vector<uint8_t> &isGood, 
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
                //std::cout << "HITrackClusterRemover::cleanup " << typeid(T).name() << " reference " << index << " to " << (refs.size() - 1) << std::endl;
            }

	}
        if (outds.empty()) outds.abort(); // not write in an empty DSV
    }

    if (oldRefs != 0) mergeOld(refs, *oldRefs);
    return output;
}

float HITrackClusterRemover::sensorThickness (const SiStripDetId& detid) const
{
  if (detid.subdetId()>=SiStripDetId::TIB) {
    if (detid.subdetId()==SiStripDetId::TOB) return 0.047;
    if (detid.moduleGeometry()==SiStripDetId::W5 || detid.moduleGeometry()==SiStripDetId::W6 ||
        detid.moduleGeometry()==SiStripDetId::W7)
	return 0.047;
    return 0.029; // so it is TEC ring 1-4 or TIB or TOB;
  } else if (detid.subdetId()==PixelSubdetector::PixelBarrel) return 0.0285;
  else return 0.027;
}

void HITrackClusterRemover::process(OmniClusterRef const & ocluster, SiStripDetId & detid, bool fromTrack) {
    SiStripRecHit2D::ClusterRef cluster = ocluster.cluster_strip();
  if (cluster.id() != stripSourceProdID) throw cms::Exception("Inconsistent Data") <<
    "HITrackClusterRemover: strip cluster ref from Product ID = " << cluster.id() <<
    " does not match with source cluster collection (ID = " << stripSourceProdID << ")\n.";
  
  uint32_t subdet = detid.subdetId();
  assert(cluster.id() == stripSourceProdID);
  if (pblocks_[subdet-1].usesSize_ && (cluster->amplitudes().size() > pblocks_[subdet-1].maxSize_)) return;
  if (!fromTrack) {
    int clusCharge=0;
    for( std::vector<uint8_t>::const_iterator iAmp = cluster->amplitudes().begin(); iAmp != cluster->amplitudes().end(); ++iAmp){
      clusCharge += *iAmp;
    }
    if (pblocks_[subdet-1].cutOnStripCharge_ && (clusCharge > (pblocks_[subdet-1].minGoodStripCharge_*sensorThickness(detid)))) return;
  }

  if (collectedStrips_.size()<=cluster.key())
    edm::LogError("BadCollectionSize")<<collectedStrips_.size()<<" is smaller than "<<cluster.key();

  assert(collectedStrips_.size() > cluster.key());
  strips[cluster.key()] = false;
  if (!clusterWasteSolution_) collectedStrips_[cluster.key()]=true;

}


void HITrackClusterRemover::process(const TrackingRecHit *hit, unsigned char chi2, const TrackerGeometry* tg) {
    SiStripDetId detid = hit->geographicalId(); 
    uint32_t subdet = detid.subdetId();

    assert ((subdet > 0) && (subdet <= NumberOfParamBlocks));

    // chi2 cut
    if (chi2 > Traj2TrackHits::toChi2x5(pblocks_[subdet-1].maxChi2_)) return;

    if(GeomDetEnumerators::isTrackerPixel(tg->geomDetSubDetector(subdet))) {
        if (!doPixel_) return;
        // this is a pixel, and i *know* it is
        const SiPixelRecHit *pixelHit = static_cast<const SiPixelRecHit *>(hit);

        SiPixelRecHit::ClusterRef cluster = pixelHit->cluster();

        if (cluster.id() != pixelSourceProdID) throw cms::Exception("Inconsistent Data") << 
                "HITrackClusterRemover: pixel cluster ref from Product ID = " << cluster.id() << 
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
            process(stripHit->omniClusterRef(),detid, true);}
	else if (hitType == typeid(SiStripRecHit1D)) {
	  const SiStripRecHit1D *hit1D = static_cast<const SiStripRecHit1D *>(hit);
	  process(hit1D->omniClusterRef(),detid, true);
        } else if (hitType == typeid(SiStripMatchedRecHit2D)) {
            const SiStripMatchedRecHit2D *matchHit = static_cast<const SiStripMatchedRecHit2D *>(hit);
//DBG//     cout << "Matched RecHit 2D: " << endl;
            process(matchHit->monoClusterRef(),detid, true);
            process(matchHit->stereoClusterRef(),detid, true);
        } else if (hitType == typeid(ProjectedSiStripRecHit2D)) {
            const ProjectedSiStripRecHit2D *projHit = static_cast<const ProjectedSiStripRecHit2D *>(hit);
//DBG//     cout << "Projected RecHit 2D: " << endl;
            process(projHit->originalHit().omniClusterRef(),detid, true);
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
HITrackClusterRemover::produce(Event& iEvent, const EventSetup& iSetup)
{
    ProductID pixelOldProdID, stripOldProdID;

    edm::ESHandle<TrackerGeometry> tgh;
    iSetup.get<TrackerDigiGeometryRecord>().get(tgh);
    
    Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
    if (doPixel_) {
        iEvent.getByToken(pixelClusters_, pixelClusters);
        pixelSourceProdID = pixelClusters.id();
    }
//DBG// std::cout << "HITrackClusterRemover: Read pixel " << pixelClusters_.encode() << " = ID " << pixelSourceProdID << std::endl;

    Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
    if (doStrip_) {
        iEvent.getByToken(stripClusters_, stripClusters);
        stripSourceProdID = stripClusters.id();
    }
//DBG// std::cout << "HITrackClusterRemover: Read strip " << stripClusters_.encode() << " = ID " << stripSourceProdID << std::endl;

    std::unique_ptr<ClusterRemovalInfo> cri;
    if (clusterWasteSolution_){
      if (doStrip_ && doPixel_) cri = std::make_unique<ClusterRemovalInfo>(pixelClusters, stripClusters);
      else if (doStrip_) cri = std::make_unique<ClusterRemovalInfo>(stripClusters);
      else if (doPixel_) cri = std::make_unique<ClusterRemovalInfo>(pixelClusters);
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
            throw cms::Exception("Inconsistent Data") << "HITrackClusterRemover: " <<
                "Input collection product IDs are [pixel: " << pixelClusters.id() << ", strip: " << stripClusters.id() << "] \n" <<
                "\t but the *old* ClusterRemovalInfo " << labels.productInstance << " refers as 'new product ids' to " <<
                    "[pixel: " << oldRemovalInfo->pixelNewRefProd().id() << ", strip: " << oldRemovalInfo->stripNewRefProd().id() << "]\n" << 
                "NOTA BENE: when running HITrackClusterRemover with an old ClusterRemovalInfo the hits in the trajectory MUST be already re-keyed.\n";
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
      LogDebug("HITrackClusterRemover")<<"to merge in, "<<oldStrMask->size()<<" strp and "<<oldPxlMask->size()<<" pxl";
      oldStrMask->copyMaskTo(collectedStrips_);
      oldPxlMask->copyMaskTo(collectedPixels_);
      assert(stripClusters->dataSize()>=collectedStrips_.size());
      collectedStrips_.resize(stripClusters->dataSize(), false);
    }else {
      collectedStrips_.resize(stripClusters->dataSize(), false);
      collectedPixels_.resize(pixelClusters->dataSize(), false);
    } 

    if (doTracks_) {

      Handle<reco::TrackCollection> tracks; 
      iEvent.getByToken(tracks_,tracks);

      std::vector<Handle<edm::ValueMap<int> > > quals;
      if ( overrideTrkQuals_.size() > 0) {
	quals.resize(1);
	iEvent.getByToken(overrideTrkQuals_[0],quals[0]);
      }
      int it=0;
      for (const auto & track: *tracks) {
	if (filterTracks_) {
	  bool goodTk = true;
	  if ( quals.size()!=0) {
	    int qual=(*(quals[0])). get(it++);
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
        auto const & chi2sX5 = track.extra()->chi2sX5();
        assert(chi2sX5.size()==track.recHitsSize());
        auto hb = track.recHitsBegin();
        for(unsigned int h=0;h<track.recHitsSize();h++){
          auto hit = *(hb+h);
	  if (!hit->isValid()) continue;
	  process( hit, chi2sX5[h], tgh.product());
	}
      }
    }

    if (doStripChargeCheck_) {
      edm::Handle<SiStripRecHit2DCollection> rechitsrphi;
      iEvent.getByToken(rphiRecHitToken_, rechitsrphi);
      const SiStripRecHit2DCollection::DataContainer * rphiRecHits = & (rechitsrphi).product()->data();
      for(  SiStripRecHit2DCollection::DataContainer::const_iterator
	      recHit = rphiRecHits->begin(); recHit!= rphiRecHits->end(); recHit++){
	SiStripDetId detid = recHit->geographicalId(); 
	process(recHit->omniClusterRef(),detid,false);
      }
      edm::Handle<SiStripRecHit2DCollection> rechitsstereo;
      iEvent.getByToken(stereoRecHitToken_, rechitsstereo);
      const SiStripRecHit2DCollection::DataContainer * stereoRecHits = & (rechitsstereo).product()->data();
      for(  SiStripRecHit2DCollection::DataContainer::const_iterator
	      recHit = stereoRecHits->begin(); recHit!= stereoRecHits->end(); recHit++){
	SiStripDetId detid = recHit->geographicalId(); 
	process(recHit->omniClusterRef(),detid,false);
      }

    }
//    if(doPixelChargeCheck_) {
//	edm::Handle<SiPixelRecHitCollection> pixelrechits;
//	iEvent.getByToken(pixelRecHitsToken_,pixelrechits);
//    }

    if (doPixel_ && clusterWasteSolution_) {
        OrphanHandle<edmNew::DetSetVector<SiPixelCluster> > newPixels =
          iEvent.put(cleanup(*pixelClusters, pixels, cri->pixelIndices(), mergeOld_ ? &oldRemovalInfo->pixelIndices() : 0));
//DBG// std::cout << "HITrackClusterRemover: Wrote pixel " << newPixels.id() << " from " << pixelSourceProdID << std::endl;
        cri->setNewPixelClusters(newPixels);
    }
    if (doStrip_ && clusterWasteSolution_) {
        OrphanHandle<edmNew::DetSetVector<SiStripCluster> > newStrips = iEvent.put(cleanup(*stripClusters, strips, cri->stripIndices(), mergeOld_ ? &oldRemovalInfo->stripIndices() : 0));
//DBG// std::cout << "HITrackClusterRemover: Wrote strip " << newStrips.id() << " from " << stripSourceProdID << std::endl;
        cri->setNewStripClusters(newStrips);
    }

    
    if (clusterWasteSolution_) {
      //      double fraction_pxl= cri->pixelIndices().size() / (double) pixels.size();
      //      double fraction_strp= cri->stripIndices().size() / (double) strips.size();
      //      edm::LogWarning("HITrackClusterRemover")<<" fraction: " << fraction_pxl <<" "<<fraction_strp;
      iEvent.put(std::move(cri));
    }

    pixels.clear(); strips.clear(); 

    if (!clusterWasteSolution_){
      //auto_ptr<edmNew::DetSetVector<SiPixelClusterRefNew> > removedPixelClsuterRefs(new edmNew::DetSetVector<SiPixelClusterRefNew>());
      //auto_ptr<edmNew::DetSetVector<SiStripRecHit1D::ClusterRef> > removedStripClsuterRefs(new edmNew::DetSetVector<SiStripRecHit1D::ClusterRef>());
      
      LogDebug("HITrackClusterRemover")<<"total strip to skip: "<<std::count(collectedStrips_.begin(),collectedStrips_.end(),true);
      // std::cout << "HITrackClusterRemover " <<"total strip to skip: "<<std::count(collectedStrips_.begin(),collectedStrips_.end(),true) <<std::endl;
      iEvent.put(std::make_unique<StripMaskContainer>(edm::RefProd<edmNew::DetSetVector<SiStripCluster> >(stripClusters),collectedStrips_));

      LogDebug("HITrackClusterRemover")<<"total pxl to skip: "<<std::count(collectedPixels_.begin(),collectedPixels_.end(),true);
      iEvent.put(std::make_unique<PixelMaskContainer>(edm::RefProd<edmNew::DetSetVector<SiPixelCluster> >(pixelClusters),collectedPixels_));
      
    }
    collectedStrips_.clear();
    collectedPixels_.clear();

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HITrackClusterRemover);
