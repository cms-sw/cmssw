#ifndef DeDxHitInfo_H
#define DeDxHitInfo_H
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class DeDxHitInfo {
  public:
    class DeDxHitInfoContainer {
    public:
      DeDxHitInfoContainer() : charge_(0.0f), pathlength_(0.0f) {}
      DeDxHitInfoContainer(const float charge, const float pathlength, const DetId& detId, const LocalPoint& pos)
          : charge_(charge), pathlength_(pathlength), detId_(detId), pos_(pos) {}

      float charge() const { return charge_; }
      float pathlength() const { return pathlength_; }
      const DetId& detId() const { return detId_; }
      const LocalPoint& pos() const { return pos_; }

    private:
      //! total cluster charge
      float charge_;
      //! path length inside a module
      float pathlength_;
      DetId detId_;
      //! hit position
      LocalPoint pos_;
    };

    typedef std::vector<DeDxHitInfo::DeDxHitInfoContainer> DeDxHitInfoContainerCollection;

  public:
    DeDxHitInfo() {}
    size_t size() const { return infos_.size(); }
    float charge(size_t i) const { return infos_[i].charge(); }
    float pathlength(size_t i) const { return infos_[i].pathlength(); }
    DetId detId(size_t i) const { return infos_[i].detId(); }
    const LocalPoint pos(size_t i) const { return infos_[i].pos(); }
    const SiPixelCluster* pixelCluster(size_t i) const {
      size_t P = 0;
      bool isPixel = false;
      bool isFirst = true;
      for (size_t j = 0; j <= i && j < infos_.size(); j++) {
        if (detId(j).subdetId() < SiStripDetId::TIB) {
          if (isFirst)
            isFirst = false;
          else
            P++;
          isPixel = true;
        } else {
          isPixel = false;
        }
      }
      if (isPixel && pixelClusters_.size() > P) {
        return &(pixelClusters_[P]);
      }
      return nullptr;
    }
    const SiStripCluster* stripCluster(size_t i) const {
      size_t S = 0;
      bool isStrip = false;
      bool isFirst = true;
      for (size_t j = 0; j <= i && j < infos_.size(); j++) {
        if (detId(j).subdetId() >= SiStripDetId::TIB) {
          if (isFirst) {
            isFirst = false;
          } else
            S++;
          isStrip = true;
        } else {
          isStrip = false;
        }
      }
      if (isStrip && stripClusters_.size() > S) {
        return &(stripClusters_[S]);
      }
      return nullptr;
    }
    const std::vector<SiStripCluster>& stripClusters() const { return stripClusters_; }
    const std::vector<SiPixelCluster>& pixelClusters() const { return pixelClusters_; }

    void addHit(const float charge,
                const float pathlength,
                const DetId& detId,
                const LocalPoint& pos,
                const SiStripCluster& stripCluster) {
      infos_.push_back(DeDxHitInfoContainer(charge, pathlength, detId, pos));
      stripClusters_.push_back(stripCluster);
    }
    void addHit(const float charge,
                const float pathlength,
                const DetId& detId,
                const LocalPoint& pos,
                const SiPixelCluster& pixelCluster) {
      infos_.push_back(DeDxHitInfoContainer(charge, pathlength, detId, pos));
      pixelClusters_.push_back(pixelCluster);
    }

  private:
    std::vector<DeDxHitInfoContainer> infos_;
    std::vector<SiStripCluster> stripClusters_;
    std::vector<SiPixelCluster> pixelClusters_;
  };

  typedef std::vector<DeDxHitInfo> DeDxHitInfoCollection;
  typedef edm::Ref<DeDxHitInfoCollection> DeDxHitInfoRef;
  typedef edm::RefProd<DeDxHitInfoCollection> DeDxHitInfoRefProd;
  typedef edm::RefVector<DeDxHitInfoCollection> DeDxHitInfoRefVector;
  typedef edm::Association<DeDxHitInfoCollection> DeDxHitInfoAss;
}  // namespace reco

#endif
