#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "CalorimeterDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace particleFlowRecHitProducer;

  template <typename CAL>
  class PFRecHitTopologyESProducer : public ESProducer {
  public:
    PFRecHitTopologyESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      geomToken_ = cc.consumes();
      if constexpr (std::is_same_v<CAL, HCAL>)
        hcalToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<typename CAL::TopologyTypeHost> produce(const typename CAL::TopologyRecordType& iRecord) {
      const auto& geom = iRecord.get(geomToken_);
      auto product = std::make_unique<typename CAL::TopologyTypeHost>(CAL::kSize, cms::alpakatools::host());
      auto view = product->view();

      const int calEnums[2] = {CAL::kSubdetectorBarrelId, CAL::kSubdetectorEndcapId};
      for (const auto subdet : calEnums) {
        // Construct topology
        //  for HCAL: using dedicated record
        //  for ECAL: from CaloGeometry (separate for barrel and endcap)
        const CaloSubdetectorGeometry* geo = geom.getSubdetectorGeometry(CAL::kDetectorId, subdet);
        const CaloSubdetectorTopology* topo;
        std::variant<EcalBarrelTopology, EcalEndcapTopology> topoVar;  // need to store ECAL topology temporarily
        if constexpr (std::is_same_v<CAL, HCAL>)
          topo = &iRecord.get(hcalToken_);
        else if (subdet == EcalSubdetector::EcalBarrel)
          topo = &topoVar.emplace<EcalBarrelTopology>(geom);
        else
          topo = &topoVar.emplace<EcalEndcapTopology>(geom);

        // Fill product
        for (auto const detId : geom.getValidDetIds(CAL::kDetectorId, subdet)) {
          const uint32_t denseId = CAL::detId2denseId(detId);
          assert(denseId < CAL::kSize);

          const GlobalPoint pos = geo->getGeometry(detId)->getPosition();
          view.positionX(denseId) = pos.x();
          view.positionY(denseId) = pos.y();
          view.positionZ(denseId) = pos.z();

          for (uint32_t n = 0; n < 8; n++) {
            uint32_t neighDetId = getNeighbourDetId(detId, n, *topo);
            if (CAL::detIdInRange(neighDetId))
              view.neighbours(denseId)(n) = CAL::detId2denseId(neighDetId);
            else
              view.neighbours(denseId)(n) = 0xffffffff;
          }
        }
      }

      // Remove neighbours that are not backward compatible (only for HCAL)
      if constexpr (std::is_same_v<CAL, HCAL>) {
        for (const auto subdet : calEnums)
          for (auto const detId : geom.getValidDetIds(CAL::kDetectorId, subdet)) {
            const uint32_t denseId = CAL::detId2denseId(detId);
            for (uint32_t n = 0; n < 8; n++) {
              const ::reco::PFRecHitsTopologyNeighbours& neighboursOfNeighbour =
                  view.neighbours(view.neighbours(denseId)[n]);
              if (std::find(neighboursOfNeighbour.begin(), neighboursOfNeighbour.end(), denseId) ==
                  neighboursOfNeighbour.end())
                view.neighbours(denseId)[n] = 0xffffffff;
            }
          }
      }

      // Print results (for debugging)
      LogDebug("PFRecHitTopologyESProducer").log([&](auto& log) {
        for (const auto subdet : calEnums)
          for (const auto detId : geom.getValidDetIds(CAL::kDetectorId, subdet)) {
            const uint32_t denseId = CAL::detId2denseId(detId);
            log.format("detId:{} denseId:{} pos:{},{},{} neighbours:{},{},{},{};{},{},{},{}\n",
                       (uint32_t)detId,
                       denseId,
                       view[denseId].positionX(),
                       view[denseId].positionY(),
                       view[denseId].positionZ(),
                       view[denseId].neighbours()(0),
                       view[denseId].neighbours()(1),
                       view[denseId].neighbours()(2),
                       view[denseId].neighbours()(3),
                       view[denseId].neighbours()(4),
                       view[denseId].neighbours()(5),
                       view[denseId].neighbours()(6),
                       view[denseId].neighbours()(7));
          }
      });

      return product;
    }

  private:
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
    edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;

    // specialised for HCAL/ECAL, because non-nearest neighbours are defined differently
    uint32_t getNeighbourDetId(const uint32_t detId, const uint32_t direction, const CaloSubdetectorTopology& topo);
  };

  template <>
  uint32_t PFRecHitTopologyESProducer<ECAL>::getNeighbourDetId(const uint32_t detId,
                                                               const uint32_t direction,
                                                               const CaloSubdetectorTopology& topo) {
    // desired order for PF: NORTH, SOUTH, EAST, WEST, NORTHEAST, SOUTHWEST, SOUTHEAST, NORTHWEST
    if (detId == 0)
      return 0;

    if (direction == 0)            // NORTH
      return topo.goNorth(detId);  // larger iphi values (except phi boundary)
    if (direction == 1)            // SOUTH
      return topo.goSouth(detId);  // smaller iphi values (except phi boundary)
    if (direction == 2)            // EAST
      return topo.goEast(detId);   // smaller ieta values
    if (direction == 3)            // WEST
      return topo.goWest(detId);   // larger ieta values

    if (direction == 4) {  // NORTHEAST
      const uint32_t NE = getNeighbourDetId(getNeighbourDetId(detId, 0, topo), 2, topo);
      if (NE)
        return NE;
      return getNeighbourDetId(getNeighbourDetId(detId, 2, topo), 0, topo);
    }
    if (direction == 5) {  // SOUTHWEST
      const uint32_t SW = getNeighbourDetId(getNeighbourDetId(detId, 1, topo), 3, topo);
      if (SW)
        return SW;
      return getNeighbourDetId(getNeighbourDetId(detId, 3, topo), 1, topo);
    }
    if (direction == 6) {  // SOUTHEAST
      const uint32_t ES = getNeighbourDetId(getNeighbourDetId(detId, 2, topo), 1, topo);
      if (ES)
        return ES;
      return getNeighbourDetId(getNeighbourDetId(detId, 1, topo), 2, topo);
    }
    if (direction == 7) {  // NORTHWEST
      const uint32_t WN = getNeighbourDetId(getNeighbourDetId(detId, 3, topo), 0, topo);
      if (WN)
        return WN;
      return getNeighbourDetId(getNeighbourDetId(detId, 0, topo), 3, topo);
    }
    return 0;
  }

  template <>
  uint32_t PFRecHitTopologyESProducer<HCAL>::getNeighbourDetId(const uint32_t detId,
                                                               const uint32_t direction,
                                                               const CaloSubdetectorTopology& topo) {
    // desired order for PF: NORTH, SOUTH, EAST, WEST, NORTHEAST, SOUTHWEST, SOUTHEAST, NORTHWEST
    if (detId == 0)
      return 0;

    if (direction == 0)            // NORTH
      return topo.goNorth(detId);  // larger iphi values (except phi boundary)
    if (direction == 1)            // SOUTH
      return topo.goSouth(detId);  // smaller iphi values (except phi boundary)
    if (direction == 2)            // EAST
      return topo.goEast(detId);   // smaller ieta values
    if (direction == 3)            // WEST
      return topo.goWest(detId);   // larger ieta values

    std::pair<uint32_t, uint32_t> directions;
    if (direction == 4) {  // NORTHEAST
      if (HCAL::getZside(detId) > 0)
        directions = {2, 0};  // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
      else
        directions = {0, 2};      // negative eta: move in phi first then move to east (coarser phi granularity)
    } else if (direction == 5) {  // SOUTHWEST
      if (HCAL::getZside(detId) > 0)
        directions = {1, 3};  // positive eta: move in phi first then move to west (coarser phi granularity)
      else
        directions = {3, 1};      // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
    } else if (direction == 6) {  // SOUTHEAST
      if (HCAL::getZside(detId) > 0)
        directions = {2, 1};  // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
      else
        directions = {1, 2};      // negative eta: move in phi first then move to east (coarser phi granularity)
    } else if (direction == 7) {  // NORTHWEST
      if (HCAL::getZside(detId) > 0)
        directions = {0, 3};  // positive eta: move in phi first then move to west (coarser phi granularity)
      else
        directions = {3, 0};  // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
    } else
      return 0;
    const uint32_t nn1 = getNeighbourDetId(detId, directions.first, topo);   // nearest neighbour in direction 1
    const uint32_t nn2 = getNeighbourDetId(detId, directions.second, topo);  // nearest neighbour in direction 2
    const uint32_t nnn = getNeighbourDetId(nn1, directions.second, topo);    // next-to-nearest neighbour
    if (nnn == nn1 || nnn == nn2)                                            // avoid duplicates
      return 0;
    return nnn;
  }

  using PFRecHitECALTopologyESProducer = PFRecHitTopologyESProducer<ECAL>;
  using PFRecHitHCALTopologyESProducer = PFRecHitTopologyESProducer<HCAL>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFRecHitECALTopologyESProducer);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFRecHitHCALTopologyESProducer);
