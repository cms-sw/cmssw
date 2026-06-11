//******************************************************************************
//
// Portable and paralelized implementation of the pixel seed matching
//
// The module produces an ElectronSeed SoA including information if a
// pixel seeds has been matched to a SC
//*******************************************************************************

#include <Eigen/Core>

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedSoA.h"
#include "DataFormats/EgammaReco/interface/SuperClusterSoA.h"
#include "DataFormats/EgammaReco/interface/alpaka/SuperClusterDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterHostCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/ElectronSeedDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedHostCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"

#include "PixelMatchingAlgo.h"

using Vector3d = Eigen::Matrix<double, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class ElectronNHitSeedAlpakaProducer : public global::EDProducer<> {
  public:
    ElectronNHitSeedAlpakaProducer(const edm::ParameterSet& pset)
        : EDProducer(pset),
          deviceToken_{produces()},
          initialSeedsToken_(consumes(pset.getParameter<edm::InputTag>("initialSeeds"))),
          beamSpotToken_(consumes(pset.getParameter<edm::InputTag>("beamSpot"))),
          superClustersTokens_(consumes(pset.getParameter<edm::InputTag>("superClusters"))) {}

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const& iSetup) const override {
      auto vprim_ = event.get(beamSpotToken_).position();
      GlobalPoint vprim(vprim_.x(), vprim_.y(), vprim_.z());
      Vector3d vertex{vprim.x(), vprim.y(), vprim.z()};

      const std::vector<reco::SuperClusterRef>& superClusterRefVec = event.get(superClustersTokens_);
      int32_t superClusterCollectionSize = superClusterRefVec.size();
      reco::SuperClusterHostCollection hostProductSCs{event.queue(), superClusterCollectionSize};

      const std::vector<TrajectorySeed>& seedVec = event.get(initialSeedsToken_);
      int32_t seedCollectionSize = seedVec.size();
      reco::ElectronSeedHostCollection hostProductSeeds{event.queue(), seedCollectionSize};

      auto& viewSCs = hostProductSCs.view();
      auto& viewSeeds = hostProductSeeds.view();

      std::cout << " -----> Collection sizes SCs: " << superClusterCollectionSize << " & Seeds " << seedCollectionSize
                << std::endl;

      int32_t i = 0;
      for (auto& superClusRef : superClusterRefVec) {
        viewSCs[i].id() = i;
        const auto& superClus = *superClusRef;
        viewSCs[i].scSeedTheta() = superClus.seed()->position().theta();
        viewSCs[i].scPhi() = superClusRef->position().phi();
        viewSCs[i].scR() = superClusRef->position().r();
        viewSCs[i].scEnergy() = superClusRef->energy();
        ++i;
      }

      // Filling in SoAs that will be copied to the GPU
      i = 0;
      for (auto& initialSeedRef : seedVec) {
        viewSeeds[i].nHits() = initialSeedRef.nHits();
        viewSeeds[i].id() = i;
        viewSeeds[i].isMatched() = 0;
        viewSeeds[i].matchedScID() = -1;

        auto hitIt = initialSeedRef.recHits().begin();

        // Hit 0
        const auto& recHit0 = *hitIt;
        const auto& pos0 = recHit0.globalPosition();
        const auto& surf0 = recHit0.det()->surface().position();
        const auto& rot0 = recHit0.det()->surface().rotation().z();
        viewSeeds[i].hit0detectorID() = (recHit0.geographicalId().subdetId() == PixelSubdetector::PixelBarrel) ? 1 : 0;
        viewSeeds[i].hit0isValid() = recHit0.isValid();
        viewSeeds[i].hit0Pos() = Eigen::Vector3d(pos0.x(), pos0.y(), pos0.z());
        viewSeeds[i].surf0Pos() = Eigen::Vector3d(surf0.x(), surf0.y(), surf0.z());
        viewSeeds[i].surf0Rot() = Eigen::Vector3d(rot0.x(), rot0.y(), rot0.z());

        // Hit 1
        ++hitIt;
        const auto& recHit1 = *hitIt;
        const auto& pos1 = recHit1.globalPosition();
        const auto& surf1 = recHit1.det()->surface().position();
        const auto& rot1 = recHit1.det()->surface().rotation().z();
        viewSeeds[i].hit1detectorID() = (recHit1.geographicalId().subdetId() == PixelSubdetector::PixelBarrel) ? 1 : 0;
        viewSeeds[i].hit1isValid() = recHit1.isValid();
        viewSeeds[i].hit1Pos() = Eigen::Vector3d(pos1.x(), pos1.y(), pos1.z());
        viewSeeds[i].surf1Pos() = Eigen::Vector3d(surf1.x(), surf1.y(), surf1.z());
        viewSeeds[i].surf1Rot() = Eigen::Vector3d(rot1.x(), rot1.y(), rot1.z());

        // Hit 2
        if (initialSeedRef.nHits() > 2) {
          ++hitIt;
          const auto& recHit2 = *hitIt;
          const auto& pos2 = recHit2.globalPosition();
          const auto& surf2 = recHit2.det()->surface().position();
          const auto& rot2 = recHit2.det()->surface().rotation().z();
          viewSeeds[i].hit2detectorID() =
              (recHit2.geographicalId().subdetId() == PixelSubdetector::PixelBarrel) ? 1 : 0;
          viewSeeds[i].hit2isValid() = recHit2.isValid();
          viewSeeds[i].hit2Pos() = Eigen::Vector3d(pos2.x(), pos2.y(), pos2.z());
          viewSeeds[i].surf2Pos() = Eigen::Vector3d(surf2.x(), surf2.y(), surf2.z());
          viewSeeds[i].surf2Rot() = Eigen::Vector3d(rot2.x(), rot2.y(), rot2.z());
        } else {
          // Zero initialization
          viewSeeds[i].hit2Pos().setZero();
          viewSeeds[i].surf2Pos().setZero();
          viewSeeds[i].surf2Rot().setZero();
          viewSeeds[i].hit2detectorID() = 0;
          viewSeeds[i].hit2isValid() = 0;
        }
        ++i;
      }

      // Create device products & copy to device
      reco::SuperClusterDeviceCollection deviceProductSCs{event.queue(), superClusterCollectionSize};
      reco::ElectronSeedDeviceCollection deviceProductSeeds{event.queue(), seedCollectionSize};
      alpaka::memcpy(event.queue(), deviceProductSCs.buffer(), hostProductSCs.buffer());
      alpaka::memcpy(event.queue(), deviceProductSeeds.buffer(), hostProductSeeds.buffer());

      algo_.matchSeeds(event.queue(), deviceProductSeeds, deviceProductSCs, vertex(0), vertex(1), vertex(2));

      event.emplace(deviceToken_, std::move(deviceProductSeeds));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("initialSeeds", {"hltElePixelSeedsCombined"});
      desc.add<edm::InputTag>("beamSpot", {"hltOnlineBeamSpot"});
      desc.add<edm::InputTag>("superClusters", {"hltEgammaSuperClustersToPixelMatch"});
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<reco::ElectronSeedDeviceCollection> deviceToken_;
    const edm::EDGetTokenT<TrajectorySeedCollection> initialSeedsToken_;
    const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    const edm::EDGetTokenT<std::vector<reco::SuperClusterRef>> superClustersTokens_;
    PixelMatchingAlgo const algo_{};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(ElectronNHitSeedAlpakaProducer);
