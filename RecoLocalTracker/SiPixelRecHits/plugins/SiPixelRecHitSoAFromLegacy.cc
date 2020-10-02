#include <cuda_runtime.h>

// hack waiting for if constexpr
#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

#include "RecoLocalTracker/SiPixelRecHits/plugins/gpuPixelRecHits.h"

class SiPixelRecHitSoAFromLegacy : public edm::global::EDProducer<> {
public:
  explicit SiPixelRecHitSoAFromLegacy(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitSoAFromLegacy() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HitModuleStart = std::array<uint32_t, gpuClustering::MaxNumModules + 1>;
  using HMSstorage = HostProduct<unsigned int[]>;

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  // The mess with inputs will be cleaned up when migrating to the new framework
  edm::EDGetTokenT<reco::BeamSpot> bsGetToken_;
  edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;  // Legacy Clusters
  edm::EDPutTokenT<TrackingRecHit2DCPU> tokenHit_;
  edm::EDPutTokenT<HMSstorage> tokenModuleStart_;

  std::string const cpeName_;
  bool const convert2Legacy_;
};

SiPixelRecHitSoAFromLegacy::SiPixelRecHitSoAFromLegacy(const edm::ParameterSet& iConfig)
    : bsGetToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))},
      clusterToken_{consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("src"))},
      tokenHit_{produces<TrackingRecHit2DCPU>()},
      tokenModuleStart_{produces<HMSstorage>()},
      cpeName_(iConfig.getParameter<std::string>("CPE")),
      convert2Legacy_(iConfig.getParameter<bool>("convertToLegacy")) {
  if (convert2Legacy_)
    produces<SiPixelRecHitCollectionNew>();
}

void SiPixelRecHitSoAFromLegacy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<std::string>("CPE", "PixelCPEFast");
  desc.add<bool>("convertToLegacy", false);
  descriptions.add("siPixelRecHitHostSoA", desc);
}

void SiPixelRecHitSoAFromLegacy::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& es) const {
  const TrackerGeometry* geom_ = nullptr;
  const PixelClusterParameterEstimator* cpe_ = nullptr;

  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get(geom);
  geom_ = geom.product();

  edm::ESHandle<PixelClusterParameterEstimator> hCPE;
  es.get<TkPixelCPERecord>().get(cpeName_, hCPE);
  cpe_ = dynamic_cast<const PixelCPEBase*>(hCPE.product());

  PixelCPEFast const* fcpe = dynamic_cast<const PixelCPEFast*>(cpe_);
  if (!fcpe) {
    throw cms::Exception("Configuration") << "too bad, not a fast cpe gpu processing not possible....";
  }
  auto const& cpeView = fcpe->getCPUProduct();

  const reco::BeamSpot& bs = iEvent.get(bsGetToken_);

  BeamSpotPOD bsHost;
  bsHost.x = bs.x0();
  bsHost.y = bs.y0();
  bsHost.z = bs.z0();

  edm::Handle<SiPixelClusterCollectionNew> hclusters;
  iEvent.getByToken(clusterToken_, hclusters);
  auto const& input = *hclusters;

  // yes a unique ptr of a unique ptr so edm is happy and the pointer stay still...
  auto hmsp = std::make_unique<uint32_t[]>(gpuClustering::MaxNumModules + 1);
  auto hitsModuleStart = hmsp.get();
  auto hms = std::make_unique<HMSstorage>(std::move(hmsp));  // hmsp is gone
  iEvent.put(tokenModuleStart_, std::move(hms));             // hms is gone! hitsModuleStart still alive and kicking...

  // legacy output
  auto legacyOutput = std::make_unique<SiPixelRecHitCollectionNew>();

  // storage
  std::vector<uint16_t> xx_;
  std::vector<uint16_t> yy_;
  std::vector<uint16_t> adc_;
  std::vector<uint16_t> moduleInd_;
  std::vector<int32_t> clus_;

  std::vector<edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster>> clusterRef;

  constexpr uint32_t MaxHitsInModule = gpuClustering::MaxHitsInModule;

  HitModuleStart moduleStart_;  // index of the first pixel of each module
  HitModuleStart clusInModule_;
  memset(&clusInModule_, 0, sizeof(HitModuleStart));  // needed??
  assert(2001 == clusInModule_.size());
  assert(0 == clusInModule_[2000]);
  uint32_t moduleId_;
  moduleStart_[1] = 0;  // we run sequentially....

  SiPixelClustersCUDA::DeviceConstView clusterView{
      moduleStart_.data(), clusInModule_.data(), &moduleId_, hitsModuleStart};

  // fill cluster arrays
  int numberOfClusters = 0;
  for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++) {
    unsigned int detid = DSViter->detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto gind = genericDet->index();
    assert(gind < 2000);
    auto const nclus = DSViter->size();
    clusInModule_[gind] = nclus;
    numberOfClusters += nclus;
  }
  hitsModuleStart[0] = 0;
  for (int i = 1, n = clusInModule_.size(); i < n; ++i)
    hitsModuleStart[i] = hitsModuleStart[i - 1] + clusInModule_[i - 1];
  assert(numberOfClusters == int(hitsModuleStart[2000]));

  // output SoA
  auto output = std::make_unique<TrackingRecHit2DCPU>(numberOfClusters, &cpeView, hitsModuleStart, nullptr);

  if (0 == numberOfClusters) {
    iEvent.put(std::move(output));
    if (convert2Legacy_)
      iEvent.put(std::move(legacyOutput));
    return;
  }

  if (convert2Legacy_)
    legacyOutput->reserve(2000, numberOfClusters);

  int numberOfDetUnits = 0;
  int numberOfHits = 0;
  for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++) {
    numberOfDetUnits++;
    unsigned int detid = DSViter->detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto const gind = genericDet->index();
    assert(gind < 2000);
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    auto const nclus = DSViter->size();
    assert(clusInModule_[gind] == nclus);
    if (0 == nclus)
      continue;  // is this really possible?

    auto const fc = hitsModuleStart[gind];
    auto const lc = hitsModuleStart[gind + 1];
    assert(lc > fc);
    // std::cout << "in det " << gind << ": conv " << nclus << " hits from " << DSViter->size() << " legacy clusters"
    //          <<' '<< fc <<','<<lc<<std::endl;
    assert((lc - fc) == nclus);
    if (nclus > MaxHitsInModule)
      printf(
          "WARNING: too many clusters %d in Module %d. Only first %d Hits converted\n", nclus, gind, MaxHitsInModule);

    // fill digis
    xx_.clear();
    yy_.clear();
    adc_.clear();
    moduleInd_.clear();
    clus_.clear();
    clusterRef.clear();
    moduleId_ = gind;
    uint32_t ic = 0;
    uint32_t ndigi = 0;
    for (auto const& clust : *DSViter) {
      assert(clust.size() > 0);
      for (int i = 0, nd = clust.size(); i < nd; ++i) {
        auto px = clust.pixel(i);
        xx_.push_back(px.x);
        yy_.push_back(px.y);
        adc_.push_back(px.adc);
        moduleInd_.push_back(gind);
        clus_.push_back(ic);
        ++ndigi;
      }
      assert(clust.originalId() == ic);  // make sure hits and clus are in sync
      if (convert2Legacy_)
        clusterRef.emplace_back(edmNew::makeRefTo(hclusters, &clust));
      ic++;
    }
    assert(nclus == ic);
    assert(clus_.size() == ndigi);
    numberOfHits += nclus;
    // filled creates view
    SiPixelDigisCUDA::DeviceConstView digiView{xx_.data(), yy_.data(), adc_.data(), moduleInd_.data(), clus_.data()};
    assert(digiView.adc(0) != 0);
    // not needed...
    cms::cudacompat::resetGrid();
    // we run on blockId.x==0
    gpuPixelRecHits::getHits(&cpeView, &bsHost, &digiView, ndigi, &clusterView, output->view());
    for (auto h = fc; h < lc; ++h)
      if (h - fc < MaxHitsInModule)
        assert(gind == output->view()->detectorIndex(h));
      else
        assert(9999 == output->view()->detectorIndex(h));
    if (convert2Legacy_) {
      SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(*legacyOutput, detid);
      for (auto h = fc; h < lc; ++h) {
        auto ih = h - fc;
        if (ih >= MaxHitsInModule)
          break;
        assert(ih < clusterRef.size());
        LocalPoint lp(output->view()->xLocal(h), output->view()->yLocal(h));
        LocalError le(output->view()->xerrLocal(h), 0, output->view()->yerrLocal(h));
        SiPixelRecHitQuality::QualWordType rqw = 0;
        SiPixelRecHit hit(lp, le, rqw, *genericDet, clusterRef[ih]);
        recHitsOnDetUnit.push_back(hit);
      }
    }
  }
  assert(numberOfHits == numberOfClusters);

  // fill data structure to support CA
  for (auto i = 0; i < 11; ++i) {
    output->hitsLayerStart()[i] = hitsModuleStart[cpeView.layerGeometry().layerStart[i]];
  }
  cms::cuda::fillManyFromVector(
      output->phiBinner(), 10, output->iphi(), output->hitsLayerStart(), numberOfHits, 256, nullptr);

  // std::cout << "created HitSoa for " <<  numberOfClusters << " clusters in " << numberOfDetUnits << " Dets" << std::endl;
  iEvent.put(std::move(output));
  if (convert2Legacy_)
    iEvent.put(std::move(legacyOutput));
}

DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacy);
