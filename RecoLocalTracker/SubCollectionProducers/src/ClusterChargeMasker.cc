#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"


#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"


namespace {

  class ClusterChargeMasker : public edm::stream::EDProducer<> {
  public:
    ClusterChargeMasker(const edm::ParameterSet& iConfig);
    ~ClusterChargeMasker(){}
    void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override ;
  private:
 

    using PixelMaskContainer = edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster>>;
    using StripMaskContainer = edm::ContainerMask<edmNew::DetSetVector<SiStripCluster>>;


    bool mergeOld_;
    float minGoodStripCharge_;

    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClusters_;

    edm::EDGetTokenT<PixelMaskContainer> oldPxlMaskToken_;
    edm::EDGetTokenT<StripMaskContainer> oldStrMaskToken_;



  };


  ClusterChargeMasker::ClusterChargeMasker(const edm::ParameterSet& iConfig) :
    mergeOld_(iConfig.exists("oldClusterRemovalInfo")),
    minGoodStripCharge_(iConfig.getParameter<double>("minGoodStripCharge"))
  {
    produces<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > >();
    produces<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > >();


    pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"));
    stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("stripClusters"));

    if (mergeOld_) {
      oldPxlMaskToken_ = consumes<PixelMaskContainer>(iConfig.getParameter<edm::InputTag>("oldClusterRemovalInfo"));
      oldStrMaskToken_ = consumes<StripMaskContainer>(iConfig.getParameter<edm::InputTag>("oldClusterRemovalInfo"));
    }

  }


  void
  ClusterChargeMasker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
      LogDebug("ClusterChargeMasker")<<"to merge in, "<<oldStrMask->size()<<" strp and "<<oldPxlMask->size()<<" pxl";
      oldStrMask->copyMaskTo(collectedStrips);
      oldPxlMask->copyMaskTo(collectedPixels);
      assert(stripClusters->dataSize()>=collectedStrips.size());
      collectedStrips.resize(stripClusters->dataSize(), false);
    }else {
      collectedStrips.resize(stripClusters->dataSize(), false);
      collectedPixels.resize(pixelClusters->dataSize(), false);
    } 

    auto const & clusters = stripClusters->data();
    for (auto const & item : stripClusters->ids()) {
	
	if (!item.isValid()) continue;  // not umpacked  (hlt only)
	
	auto detid = item.id;
	
	for (auto i = item.offset; i<item.offset+int(item.size); ++i) {
          auto clusCharge = siStripClusterTools::chargePerCM(detid,clusters[i]);
	  if(clusCharge < minGoodStripCharge_) collectedStrips[i] = true; 
	}

    }

    std::auto_ptr<StripMaskContainer> removedStripClusterMask(
         new StripMaskContainer(edm::RefProd<edmNew::DetSetVector<SiStripCluster> >(stripClusters),collectedStrips));
      LogDebug("ClusterChargeMasker")<<"total strip to skip: "<<std::count(collectedStrips.begin(),collectedStrips.end(),true);
      // std::cout << "ClusterChargeMasker " <<"total strip to skip: "<<std::count(collectedStrips_.begin(),collectedStrips_.end(),true) <<std::endl;
       iEvent.put( removedStripClusterMask );

      std::auto_ptr<PixelMaskContainer> removedPixelClusterMask(
         new PixelMaskContainer(edm::RefProd<edmNew::DetSetVector<SiPixelCluster> >(pixelClusters),collectedPixels));      
      LogDebug("ClusterChargeMasker")<<"total pxl to skip: "<<std::count(collectedPixels.begin(),collectedPixels.end(),true);
      iEvent.put( removedPixelClusterMask );
 


  }


}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterChargeMasker);

