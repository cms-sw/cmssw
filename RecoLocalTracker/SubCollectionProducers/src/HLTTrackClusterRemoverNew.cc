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
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

//
// class decleration
//

class HLTTrackClusterRemoverNew final : public edm::stream::EDProducer<> {
public:
  HLTTrackClusterRemoverNew(const edm::ParameterSet &iConfig);
  ~HLTTrackClusterRemoverNew() override;
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

private:
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> const tTrackerGeom_;
  struct ParamBlock {
    ParamBlock() : isSet_(false), usesCharge_(false) {}
    ParamBlock(const edm::ParameterSet &iConfig)
        : isSet_(true),
          usesCharge_(iConfig.exists("maxCharge")),
          usesSize_(iConfig.exists("maxSize")),
          cutOnPixelCharge_(iConfig.exists("minGoodPixelCharge")),
          cutOnStripCharge_(iConfig.exists("minGoodStripCharge")),
          maxChi2_(iConfig.getParameter<double>("maxChi2")),
          maxCharge_(usesCharge_ ? iConfig.getParameter<double>("maxCharge") : 0),
          minGoodPixelCharge_(cutOnPixelCharge_ ? iConfig.getParameter<double>("minGoodPixelCharge") : 0),
          minGoodStripCharge_(cutOnStripCharge_ ? iConfig.getParameter<double>("minGoodStripCharge") : 0),
          maxSize_(usesSize_ ? iConfig.getParameter<uint32_t>("maxSize") : 0) {}
    bool isSet_, usesCharge_, usesSize_, cutOnPixelCharge_, cutOnStripCharge_;
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
  edm::EDGetTokenT<PixelMaskContainer> oldPxlMaskToken_;
  edm::EDGetTokenT<StripMaskContainer> oldStrMaskToken_;
  edm::EDGetTokenT<std::vector<Trajectory> > trajectories_;

  ParamBlock pblocks_[NumberOfParamBlocks];
  void readPSet(const edm::ParameterSet &iConfig,
                const std::string &name,
                int id1 = -1,
                int id2 = -1,
                int id3 = -1,
                int id4 = -1,
                int id5 = -1,
                int id6 = -1);

  std::vector<uint8_t> pixels, strips;                  // avoid unneed alloc/dealloc of this
  edm::ProductID pixelSourceProdID, stripSourceProdID;  // ProdIDs refs must point to (for consistency tests)

  inline void process(const TrackingRecHit *hit, float chi2, const TrackerGeometry *tg);
  inline void process(const OmniClusterRef &cluRef, uint32_t subdet);

  template <typename T>
  std::unique_ptr<edmNew::DetSetVector<T> > cleanup(const edmNew::DetSetVector<T> &oldClusters,
                                                    const std::vector<uint8_t> &isGood,
                                                    reco::ClusterRemovalInfo::Indices &refs,
                                                    const reco::ClusterRemovalInfo::Indices *oldRefs);

  // Carries in full removal info about a given det from oldRefs
  void mergeOld(reco::ClusterRemovalInfo::Indices &refs, const reco::ClusterRemovalInfo::Indices &oldRefs);

  bool makeProducts_;
  bool doStripChargeCheck_, doPixelChargeCheck_;
  std::vector<bool> collectedRegStrips_;
  std::vector<bool> collectedPixels_;
};

using namespace std;
using namespace edm;
using namespace reco;

void HLTTrackClusterRemoverNew::readPSet(
    const edm::ParameterSet &iConfig, const std::string &name, int id1, int id2, int id3, int id4, int id5, int id6) {
  if (iConfig.exists(name)) {
    ParamBlock pblock(iConfig.getParameter<ParameterSet>(name));
    if (id1 == -1) {
      fill(pblocks_, pblocks_ + NumberOfParamBlocks, pblock);
    } else {
      pblocks_[id1] = pblock;
      if (id2 != -1)
        pblocks_[id2] = pblock;
      if (id3 != -1)
        pblocks_[id3] = pblock;
      if (id4 != -1)
        pblocks_[id4] = pblock;
      if (id5 != -1)
        pblocks_[id5] = pblock;
      if (id6 != -1)
        pblocks_[id6] = pblock;
    }
  }
}

HLTTrackClusterRemoverNew::HLTTrackClusterRemoverNew(const ParameterSet &iConfig)
    : tTrackerGeom_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      doTracks_(iConfig.exists("trajectories")),
      doStrip_(iConfig.existsAs<bool>("doStrip") ? iConfig.getParameter<bool>("doStrip") : true),
      doPixel_(iConfig.existsAs<bool>("doPixel") ? iConfig.getParameter<bool>("doPixel") : true),
      mergeOld_(false),
      makeProducts_(true),
      doStripChargeCheck_(
          iConfig.existsAs<bool>("doStripChargeCheck") ? iConfig.getParameter<bool>("doStripChargeCheck") : false),
      doPixelChargeCheck_(
          iConfig.existsAs<bool>("doPixelChargeCheck") ? iConfig.getParameter<bool>("doPixelChargeCheck") : false)

{
  if (iConfig.exists("oldClusterRemovalInfo")) {
    oldPxlMaskToken_ = consumes<PixelMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
    oldStrMaskToken_ = consumes<StripMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
    if (not(iConfig.getParameter<InputTag>("oldClusterRemovalInfo") == edm::InputTag()))
      mergeOld_ = true;
  }

  if ((doPixelChargeCheck_ && !doPixel_) || (doStripChargeCheck_ && !doStrip_))
    throw cms::Exception("Configuration Error")
        << "HLTTrackClusterRemoverNew: Charge check asked without cluster collection ";
  if (doPixelChargeCheck_)
    throw cms::Exception("Configuration Error")
        << "HLTTrackClusterRemoverNew: Pixel cluster charge check not yet implemented";

  fill(pblocks_, pblocks_ + NumberOfParamBlocks, ParamBlock());
  readPSet(iConfig, "Common", -1);
  if (doPixel_) {
    readPSet(iConfig, "Pixel", 0, 1);
    readPSet(iConfig, "PXB", 0);
    readPSet(iConfig, "PXE", 1);
  }
  if (doStrip_) {
    readPSet(iConfig, "Strip", 2, 3, 4, 5);
    readPSet(iConfig, "StripInner", 2, 3);
    readPSet(iConfig, "StripOuter", 4, 5);
    readPSet(iConfig, "TIB", 2);
    readPSet(iConfig, "TID", 3);
    readPSet(iConfig, "TOB", 4);
    readPSet(iConfig, "TEC", 5);
  }

  bool usingCharge = false;
  for (size_t i = 0; i < NumberOfParamBlocks; ++i) {
    if (!pblocks_[i].isSet_)
      throw cms::Exception("Configuration Error")
          << "HLTTrackClusterRemoverNew: Missing configuration for detector with subDetID = " << (i + 1);
    if (pblocks_[i].usesCharge_ && !usingCharge) {
      throw cms::Exception("Configuration Error")
          << "HLTTrackClusterRemoverNew: Configuration for subDetID = " << (i + 1)
          << " uses cluster charge, which is not enabled.";
    }
  }

  //    trajectories_ = consumes<vector<Trajectory> >(iConfig.getParameter<InputTag>("trajectories"));
  if (doTracks_)
    trajectories_ = consumes<vector<Trajectory> >(iConfig.getParameter<InputTag>("trajectories"));
  if (doPixel_)
    pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<InputTag>("pixelClusters"));
  if (doStrip_)
    stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<InputTag>("stripClusters"));
  if (mergeOld_) {
    oldPxlMaskToken_ = consumes<PixelMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
    oldStrMaskToken_ = consumes<StripMaskContainer>(iConfig.getParameter<InputTag>("oldClusterRemovalInfo"));
  }

  //produces<edmNew::DetSetVector<SiPixelClusterRefNew> >();
  //produces<edmNew::DetSetVector<SiStripRecHit1D::ClusterRegionalRef> >();

  produces<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > >();
  produces<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > >();
}

HLTTrackClusterRemoverNew::~HLTTrackClusterRemoverNew() {}

void HLTTrackClusterRemoverNew::mergeOld(ClusterRemovalInfo::Indices &refs,
                                         const ClusterRemovalInfo::Indices &oldRefs) {
  for (size_t i = 0, n = refs.size(); i < n; ++i) {
    refs[i] = oldRefs[refs[i]];
  }
}

template <typename T>
std::unique_ptr<edmNew::DetSetVector<T> > HLTTrackClusterRemoverNew::cleanup(
    const edmNew::DetSetVector<T> &oldClusters,
    const std::vector<uint8_t> &isGood,
    reco::ClusterRemovalInfo::Indices &refs,
    const reco::ClusterRemovalInfo::Indices *oldRefs) {
  typedef typename edmNew::DetSetVector<T> DSV;
  typedef typename edmNew::DetSetVector<T>::FastFiller DSF;
  typedef typename edmNew::DetSet<T> DS;
  auto output = std::make_unique<DSV>();
  output->reserve(oldClusters.size(), oldClusters.dataSize());

  // cluster removal loop
  const T *firstOffset = &oldClusters.data().front();
  for (typename DSV::const_iterator itdet = oldClusters.begin(), enddet = oldClusters.end(); itdet != enddet; ++itdet) {
    DS oldDS = *itdet;

    if (oldDS.empty())
      continue;  // skip empty detsets

    uint32_t id = oldDS.detId();
    DSF outds(*output, id);

    for (typename DS::const_iterator it = oldDS.begin(), ed = oldDS.end(); it != ed; ++it) {
      uint32_t index = ((&*it) - firstOffset);
      if (isGood[index]) {
        outds.push_back(*it);
        refs.push_back(index);
        //std::cout << "HLTTrackClusterRemoverNew::cleanup " << typeid(T).name() << " reference " << index << " to " << (refs.size() - 1) << std::endl;
      }
    }
    if (outds.empty())
      outds.abort();  // not write in an empty DSV
  }
  //    std::cout<<"fraction: "<<fraction<<std::endl;
  if (oldRefs != nullptr)
    mergeOld(refs, *oldRefs);
  return output;
}

void HLTTrackClusterRemoverNew::process(OmniClusterRef const &clusterReg, uint32_t subdet) {
  if (clusterReg.id() != stripSourceProdID)
    throw cms::Exception("Inconsistent Data")
        << "HLTTrackClusterRemoverNew: strip cluster ref from Product ID = " << clusterReg.id()
        << " does not match with source cluster collection (ID = " << stripSourceProdID << ")\n.";

  if (collectedRegStrips_.size() <= clusterReg.key()) {
    edm::LogError("BadCollectionSize") << collectedRegStrips_.size() << " is smaller than " << clusterReg.key();

    assert(collectedRegStrips_.size() > clusterReg.key());
  }
  collectedRegStrips_[clusterReg.key()] = true;
}

void HLTTrackClusterRemoverNew::process(const TrackingRecHit *hit, float chi2, const TrackerGeometry *tg) {
  DetId detid = hit->geographicalId();
  uint32_t subdet = detid.subdetId();

  assert((subdet > 0) && (subdet <= NumberOfParamBlocks));

  // chi2 cut
  if (chi2 > pblocks_[subdet - 1].maxChi2_)
    return;

  if (GeomDetEnumerators::isTrackerPixel(tg->geomDetSubDetector(subdet))) {
    //      std::cout<<"process pxl hit"<<std::endl;
    if (!doPixel_)
      return;
    // this is a pixel, and i *know* it is
    const SiPixelRecHit *pixelHit = static_cast<const SiPixelRecHit *>(hit);

    SiPixelRecHit::ClusterRef cluster = pixelHit->cluster();
    if (cluster.id() != pixelSourceProdID)
      throw cms::Exception("Inconsistent Data")
          << "HLTTrackClusterRemoverNew: pixel cluster ref from Product ID = " << cluster.id()
          << " does not match with source cluster collection (ID = " << pixelSourceProdID << ")\n.";

    assert(cluster.id() == pixelSourceProdID);
    //DBG// cout << "HIT NEW PIXEL DETID = " << detid.rawId() << ", Cluster [ " << cluster.key().first << " / " <<  cluster.key().second << " ] " << endl;

    // if requested, cut on cluster size
    if (pblocks_[subdet - 1].usesSize_ && (cluster->pixels().size() > pblocks_[subdet - 1].maxSize_))
      return;

    // mark as used
    //pixels[cluster.key()] = false;
    assert(collectedPixels_.size() > cluster.key());
    collectedPixels_[cluster.key()] = true;
  } else {  // aka Strip
    if (!doStrip_)
      return;
    const type_info &hitType = typeid(*hit);
    if (hitType == typeid(SiStripRecHit2D)) {
      const SiStripRecHit2D *stripHit = static_cast<const SiStripRecHit2D *>(hit);
      //DBG//     cout << "Plain RecHit 2D: " << endl;
      process(stripHit->omniClusterRef(), subdet);
      //	  int clusCharge=0;
      //	  for ( auto cAmp : stripHit->omniClusterRef().stripCluster().amplitudes() ) clusCharge+=cAmp;
      //	  std::cout << "[HLTTrackClusterRemoverNew::process (SiStripRecHit2D) chi2: " << chi2 << " [" << subdet << " --> charge: " << clusCharge << "]" << std::endl;
    } else if (hitType == typeid(SiStripRecHit1D)) {
      const SiStripRecHit1D *hit1D = static_cast<const SiStripRecHit1D *>(hit);
      process(hit1D->omniClusterRef(), subdet);
      //	  int clusCharge=0;
      //	  for ( auto cAmp : hit1D->omniClusterRef().stripCluster().amplitudes() ) clusCharge+=cAmp;
      //	  std::cout << "[HLTTrackClusterRemoverNew::process (SiStripRecHit1D) chi2: " << chi2 << " [" << subdet << " --> charge: " << clusCharge << "]" << std::endl;
    } else if (hitType == typeid(SiStripMatchedRecHit2D)) {
      const SiStripMatchedRecHit2D *matchHit = static_cast<const SiStripMatchedRecHit2D *>(hit);
      //DBG//     cout << "Matched RecHit 2D: " << endl;
      process(matchHit->monoClusterRef(), subdet);
      //	    int clusCharge=0;
      //	    for ( auto cAmp : matchHit->monoClusterRef().stripCluster().amplitudes() ) clusCharge+=cAmp;
      //	    std::cout << "[HLTTrackClusterRemoverNew::process (SiStripMatchedRecHit2D:mono) chi2: " << chi2 << " [" << subdet << " --> charge: " << clusCharge << "]" << std::endl;

      process(matchHit->stereoClusterRef(), subdet);
      //	    clusCharge=0;
      //	    for ( auto cAmp : matchHit->stereoClusterRef().stripCluster().amplitudes() ) clusCharge+=cAmp;
      //	    std::cout << "[HLTTrackClusterRemoverNew::process (SiStripMatchedRecHit2D:stereo) chi2: " << chi2 << " [" << subdet << " --> charge: " << clusCharge << "]" << std::endl;

    } else if (hitType == typeid(ProjectedSiStripRecHit2D)) {
      const ProjectedSiStripRecHit2D *projHit = static_cast<const ProjectedSiStripRecHit2D *>(hit);
      //DBG//     cout << "Projected RecHit 2D: " << endl;
      process(projHit->originalHit().omniClusterRef(), subdet);
      //	    int clusCharge=0;
      //	    for ( auto cAmp : projHit->originalHit().omniClusterRef().stripCluster().amplitudes() ) clusCharge+=cAmp;
      //	    std::cout << "[HLTTrackClusterRemoverNew::process (ProjectedSiStripRecHit2D) chi2: " << chi2 << " [" << subdet << " --> charge: " << clusCharge << "]" << std::endl;
    } else
      throw cms::Exception("NOT IMPLEMENTED")
          << "Don't know how to handle " << hitType.name() << " on detid " << detid.rawId() << "\n";
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

void HLTTrackClusterRemoverNew::produce(Event &iEvent, const EventSetup &iSetup) {
  ProductID pixelOldProdID, stripOldProdID;

  const auto &tgh = &iSetup.getData(tTrackerGeom_);

  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
  if (doPixel_) {
    iEvent.getByToken(pixelClusters_, pixelClusters);
    pixelSourceProdID = pixelClusters.id();
  }

  edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
  if (doStrip_) {
    iEvent.getByToken(stripClusters_, stripClusters);
    stripSourceProdID = stripClusters.id();
  }

  //Handle<TrajTrackAssociationCollection> trajectories;
  edm::Handle<vector<Trajectory> > trajectories;
  iEvent.getByToken(trajectories_, trajectories);

  if (mergeOld_) {
    edm::Handle<PixelMaskContainer> oldPxlMask;
    edm::Handle<StripMaskContainer> oldStrMask;
    iEvent.getByToken(oldPxlMaskToken_, oldPxlMask);
    iEvent.getByToken(oldStrMaskToken_, oldStrMask);
    LogDebug("TrackClusterRemover") << "to merge in, " << oldStrMask->size() << " strp and " << oldPxlMask->size()
                                    << " pxl";
    oldStrMask->copyMaskTo(collectedRegStrips_);
    oldPxlMask->copyMaskTo(collectedPixels_);
    collectedRegStrips_.resize(stripClusters->dataSize(), false);
  } else {
    collectedRegStrips_.resize(stripClusters->dataSize(), false);
    collectedPixels_.resize(pixelClusters->dataSize(), false);
  }

  //for (TrajTrackAssociationCollection::const_iterator it = trajectories->begin(), ed = trajectories->end(); it != ed; ++it)  {
  //    const Trajectory &tj = * it->key;

  for (std::vector<Trajectory>::const_iterator it = trajectories->begin(), ed = trajectories->end(); it != ed; ++it) {
    const Trajectory &tj = *it;
    const std::vector<TrajectoryMeasurement> &tms = tj.measurements();

    std::vector<TrajectoryMeasurement>::const_iterator itm, endtm;
    for (itm = tms.begin(), endtm = tms.end(); itm != endtm; ++itm) {
      const TrackingRecHit *hit = itm->recHit()->hit();
      if (!hit->isValid())
        continue;
      //	    std::cout<<"process hit"<<std::endl;
      process(hit, itm->estimate(), tgh);
    }
  }

  //    std::cout << " => collectedRegStrips_: " << collectedRegStrips_.size() << std::endl;
  //    std::cout << " total strip to skip (before charge check): "<<std::count(collectedRegStrips_.begin(),collectedRegStrips_.end(),true) << std::endl;

  // checks only cluster already found! creates regression
  if (doStripChargeCheck_) {
    //      std::cout << "[HLTTrackClusterRemoverNew::produce] doStripChargeCheck_: " << (doStripChargeCheck_ ? "true" : "false") << " stripClusters: " << stripClusters->size() << std::endl;

    auto const &clusters = stripClusters->data();
    for (auto const &item : stripClusters->ids()) {
      if (!item.isValid())
        continue;  // not umpacked

      DetId detid = item.id;
      uint32_t subdet = detid.subdetId();
      if (!pblocks_[subdet - 1].cutOnStripCharge_)
        continue;

      //	std::cout << " i: " << i << " --> detid: " << detid << " --> subdet: " << subdet << std::endl;

      for (int i = item.offset; i < item.offset + int(item.size); ++i) {
        int clusCharge = 0;
        for (auto cAmp : clusters[i].amplitudes())
          clusCharge += cAmp;

        //	if (clusCharge < pblocks_[subdet-1].minGoodStripCharge_) std::cout << " clusCharge: " << clusCharge << std::endl;
        if (clusCharge < pblocks_[subdet - 1].minGoodStripCharge_)
          collectedRegStrips_[i] = true;  // (|= does not work!)
      }
    }
  }

  //    std::cout << " => collectedRegStrips_: " << collectedRegStrips_.size() << std::endl;
  //    std::cout << " total strip to skip: "<<std::count(collectedRegStrips_.begin(),collectedRegStrips_.end(),true) << std::endl;
  LogDebug("TrackClusterRemover") << "total strip to skip: "
                                  << std::count(collectedRegStrips_.begin(), collectedRegStrips_.end(), true);
  // std::cout << "TrackClusterRemover" <<" total strip to skip: "<<std::count(collectedRegStrips_.begin(),collectedRegStrips_.end(),true)<<std::endl;
  iEvent.put(std::make_unique<StripMaskContainer>(edm::RefProd<edmNew::DetSetVector<SiStripCluster> >(stripClusters),
                                                  collectedRegStrips_));

  LogDebug("TrackClusterRemover") << "total pxl to skip: "
                                  << std::count(collectedPixels_.begin(), collectedPixels_.end(), true);
  iEvent.put(std::make_unique<PixelMaskContainer>(edm::RefProd<edmNew::DetSetVector<SiPixelCluster> >(pixelClusters),
                                                  collectedPixels_));

  collectedRegStrips_.clear();
  collectedPixels_.clear();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTTrackClusterRemoverNew);
