#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigisSoA.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

namespace {
  struct AccretionCluster {
    typedef unsigned short UShort;
    static constexpr UShort MAXSIZE = 256;
    UShort adc[MAXSIZE];
    UShort x[MAXSIZE];
    UShort y[MAXSIZE];
    UShort xmin=16000;
    UShort ymin=16000;
    unsigned int isize=0;
    int charge=0;

    void clear() {
      isize=0;
      charge=0;
      xmin=16000;
      ymin=16000;
    }

    bool add(SiPixelCluster::PixelPos const & p, UShort const iadc) {
      if (isize==MAXSIZE) return false;
      xmin=std::min(xmin,(unsigned short)(p.row()));
      ymin=std::min(ymin,(unsigned short)(p.col()));
      adc[isize]=iadc;
      x[isize]=p.row();
      y[isize++]=p.col();
      charge+=iadc;
      return true;
    }
  };

  constexpr uint32_t dummydetid = 0xffffffff;
}

class SiPixelDigisClustersFromSoA: public edm::global::EDProducer<> {
public:
  explicit SiPixelDigisClustersFromSoA(const edm::ParameterSet& iConfig);
  ~SiPixelDigisClustersFromSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<SiPixelDigisSoA> digiGetToken_;

  edm::EDPutTokenT<edm::DetSetVector<PixelDigi>> digiPutToken_;
  edm::EDPutTokenT<SiPixelClusterCollectionNew> clusterPutToken_;
 
};

SiPixelDigisClustersFromSoA::SiPixelDigisClustersFromSoA(const edm::ParameterSet& iConfig):
  digiGetToken_(consumes<SiPixelDigisSoA>(iConfig.getParameter<edm::InputTag>("src"))),
  digiPutToken_(produces<edm::DetSetVector<PixelDigi>>()),
  clusterPutToken_(produces<SiPixelClusterCollectionNew>())
{}

void SiPixelDigisClustersFromSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelDigisSoA"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigisClustersFromSoA::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& digis = iEvent.get(digiGetToken_);
  
  edm::ESHandle<TrackerTopology> trackerTopologyHandle;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
  const auto& ttopo = *trackerTopologyHandle;

  auto collection = std::make_unique<edm::DetSetVector<PixelDigi>>();
  auto outputClusters = std::make_unique<SiPixelClusterCollectionNew>();

  const uint32_t nDigis = digis.size();
  edm::DetSet<PixelDigi> * detDigis=nullptr;
  for (uint32_t i = 0; i < nDigis; i++) {
    if (digis.pdigi(i)==0) continue;
    detDigis = &collection->find_or_insert(digis.rawIdArr(i));
    if ( (*detDigis).empty() ) (*detDigis).data.reserve(32); // avoid the first relocations
    break;
  }

  int32_t nclus=-1;
  std::vector<AccretionCluster> aclusters(1024);
  auto totCluseFilled=0;

  auto fillClusters = [&](uint32_t detId){
    if (nclus<0) return; // this in reality should never happen
    edmNew::DetSetVector<SiPixelCluster>::FastFiller spc(*outputClusters, detId);
    auto layer = (DetId(detId).subdetId()==1) ? ttopo.pxbLayer(detId) : 0;
    auto clusterThreshold = (layer==1) ? 2000 : 4000;
    for (int32_t ic=0; ic<nclus+1;++ic) {
      auto const & acluster = aclusters[ic];
      if ( acluster.charge < clusterThreshold) continue;
      SiPixelCluster cluster(acluster.isize,acluster.adc, acluster.x,acluster.y, acluster.xmin,acluster.ymin);
      ++totCluseFilled;
      // std::cout << "putting in this cluster " << ic << " " << cluster.charge() << " " << cluster.pixelADC().size() << endl;
      // sort by row (x)
      spc.push_back( std::move(cluster) );
      std::push_heap(spc.begin(),spc.end(),[](SiPixelCluster const & cl1,SiPixelCluster const & cl2) { return cl1.minPixelRow() < cl2.minPixelRow();});
    }
    for (int32_t ic=0; ic<nclus+1;++ic) aclusters[ic].clear();
    nclus = -1;
    // sort by row (x)
    std::sort_heap(spc.begin(),spc.end(),[](SiPixelCluster const & cl1,SiPixelCluster const & cl2) { return cl1.minPixelRow() < cl2.minPixelRow();});
    if ( spc.empty() ) spc.abort();
  };

  for (uint32_t i = 0; i < nDigis; i++) {
    if (digis.pdigi(i)==0) continue;
    if (digis.clus(i)>9000) continue; // not in cluster; TODO add an assert for the size
    assert(digis.rawIdArr(i) > 109999);
    if ( (*detDigis).detId() != digis.rawIdArr(i))
      {
        fillClusters((*detDigis).detId());
        assert(nclus==-1);
        detDigis = &collection->find_or_insert(digis.rawIdArr(i));
        if ( (*detDigis).empty() )
          (*detDigis).data.reserve(32); // avoid the first relocations
        else { std::cout << "Problem det present twice in input! " << (*detDigis).detId() << std::endl; }
      }
    (*detDigis).data.emplace_back(digis.pdigi(i));
    auto const & dig = (*detDigis).data.back();
    // fill clusters
    assert(digis.clus(i)>=0);
    assert(digis.clus(i)<1024);
    nclus = std::max(digis.clus(i),nclus);
    auto row = dig.row();
    auto col = dig.column();
    SiPixelCluster::PixelPos pix(row,col);
    aclusters[digis.clus(i)].add(pix, digis.adc(i));
  }

  // fill final clusters
  fillClusters((*detDigis).detId());
  //std::cout << "filled " << totCluseFilled << " clusters" << std::endl;

  iEvent.put(digiPutToken_, std::move(collection));
  iEvent.put(clusterPutToken_, std::move(outputClusters));
}

DEFINE_FWK_MODULE(SiPixelDigisClustersFromSoA);
