#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"

#include "SiStripClustersToLegacyAlgo.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;
  
  class SiStripClustersToLegacy2 : public stream::SynchronizingEDProducer<> {
    public:
    SiStripClustersToLegacy2(edm::ParameterSet const& iConfig);
    ~SiStripClustersToLegacy2() override = default;
    
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
    private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    void dumpClusters(edmNew::DetSetVector<SiStripCluster>* detSetClusters, int clustersPrealloc) const;
    
    private:
    const edm::EDGetTokenT<SiStripClustersDevice> siStripClustersToken_;
    const edm::EDGetTokenT<SiStripDigiHost> siStripDigiToken_;
    const edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> siStripClustersSetVecPutToken_;

    SiStripClustersToLegacyAlgo algo_;
  };
}


namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  SiStripClustersToLegacy2::SiStripClustersToLegacy2(edm::ParameterSet const& iConfig)
    : SynchronizingEDProducer(iConfig),
    siStripClustersToken_(consumes(iConfig.getParameter<edm::InputTag>("source"))),
    siStripDigiToken_(consumes(iConfig.getParameter<edm::InputTag>("source"))),
    siStripClustersSetVecPutToken_(produces()) {
    }



    void SiStripClustersToLegacy2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }


    void SiStripClustersToLegacy2::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
      const auto& clusters_d = iEvent.get(siStripClustersToken_);
      const auto& amplitudes_h = iEvent.get(siStripDigiToken_);

      const uint32_t goodCandidates = amplitudes_h->nbGoodCandidates();
      algo_.consumeSoA(iEvent.queue(), clusters_d, goodCandidates);
    }


    void SiStripClustersToLegacy2::produce(device::Event& iEvent, device::EventSetup const&) {
      auto result = algo_.convert(iEvent.queue(), iEvent.get(siStripDigiToken_));

      iEvent.put(siStripClustersSetVecPutToken_, std::move(result));
    }  
}  // namespace sistrip

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(sistrip::SiStripClustersToLegacy2);
