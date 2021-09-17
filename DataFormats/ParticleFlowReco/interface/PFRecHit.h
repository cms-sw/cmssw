#ifndef ParticleFlowReco_PFRecHit_h
#define ParticleFlowReco_PFRecHit_h
/** 
 */
#include <vector>
#include <map>
#include <memory>
#include <iostream>

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

namespace reco {

  /**\class PFRecHit
     \brief Particle flow rechit (rechit + geometry and topology information). See clustering algorithm in PFClusterAlgo
          
     \author Colin Bernet
     \date   July 2006

     Feb 2014 [Michalis: 8 years later!Modifying the class to be able to generalize the neighbours for 3D calorimeters ]
  */
  class PFRecHit {
  public:
    using PositionType = GlobalPoint::BasicVectorType;
    using REPPoint = RhoEtaPhi;
    using RepCorners = CaloCellGeometry::RepCorners;
    using REPPointVector = RepCorners;
    using CornersVec = CaloCellGeometry::CornersVec;

    struct Neighbours {
      using Pointer = unsigned int const*;
      Neighbours() {}
      Neighbours(Pointer ib, unsigned int n) : b(ib), e(ib + n) {}
      Pointer b, e;
      Pointer begin() const { return b; }
      Pointer end() const { return e; }
      unsigned int size() const { return e - b; }
    };

    enum { NONE = 0 };
    /// default constructor. Sets energy and position to zero
    PFRecHit() {}

    PFRecHit(std::shared_ptr<const CaloCellGeometry> caloCell,
             unsigned int detId,
             PFLayer::Layer layer,
             float energy,
             uint32_t flags = 0)
        : caloCell_(std::move(caloCell)), detId_(detId), layer_(layer), energy_(energy), flags_(flags) {}

    /// copy
    PFRecHit(const PFRecHit& other) = default;
    PFRecHit(PFRecHit&& other) = default;
    PFRecHit& operator=(const PFRecHit& other) = default;
    PFRecHit& operator=(PFRecHit&& other) = default;

    /// destructor
    ~PFRecHit() = default;

    void setEnergy(float energy) { energy_ = energy; }

    void addNeighbour(short x, short y, short z, unsigned int);
    unsigned int getNeighbour(short x, short y, short z) const;
    void setTime(double time) { time_ = time; }
    void setDepth(int depth) { depth_ = depth; }
    void clearNeighbours() {
      neighbours_.clear();
      neighbourInfos_.clear();
      neighbours4_ = neighbours8_ = 0;
    }

    Neighbours neighbours4() const { return buildNeighbours(neighbours4_); }
    Neighbours neighbours8() const { return buildNeighbours(neighbours8_); }

    Neighbours neighbours() const { return buildNeighbours(neighbours_.size()); }

    const std::vector<unsigned short>& neighbourInfos() { return neighbourInfos_; }

    /// calo cell
    CaloCellGeometry const& caloCell() const { return *(caloCell_.get()); }
    bool hasCaloCell() const { return (caloCell_ != nullptr); }

    /// rechit detId
    unsigned detId() const { return detId_; }

    /// rechit layer
    PFLayer::Layer layer() const { return layer_; }

    /// rechit energy
    float energy() const { return energy_; }

    /// timing for cleaned hits
    float time() const { return time_; }

    /// depth for segemntation
    int depth() const { return depth_; }

    /// rechit momentum transverse to the beam, squared.
    double pt2() const { return energy_ * energy_ * (position().perp2() / position().mag2()); }

    // Detector-dependent status flag
    uint32_t flags() const { return flags_; }

    //
    void setFlags(uint32_t flags) { flags_ = flags; }

    /// rechit cell centre x, y, z
    PositionType const& position() const { return caloCell().getPosition().basicVector(); }

    RhoEtaPhi const& positionREP() const { return caloCell().repPos(); }

    /// rechit corners
    CornersVec const& getCornersXYZ() const { return caloCell().getCorners(); }

    RepCorners const& getCornersREP() const { return caloCell().getCornersREP(); }

    /// comparison >= operator
    bool operator>=(const PFRecHit& rhs) const { return (energy_ >= rhs.energy_); }

    /// comparison > operator
    bool operator>(const PFRecHit& rhs) const { return (energy_ > rhs.energy_); }

    /// comparison <= operator
    bool operator<=(const PFRecHit& rhs) const { return (energy_ <= rhs.energy_); }

    /// comparison < operator
    bool operator<(const PFRecHit& rhs) const { return (energy_ < rhs.energy_); }

  private:
    Neighbours buildNeighbours(unsigned int n) const { return Neighbours(neighbours_.data(), n); }

    /// cell geometry
    std::shared_ptr<const CaloCellGeometry> caloCell_ = nullptr;

    ///cell detid
    unsigned int detId_ = 0;

    /// rechit layer
    PFLayer::Layer layer_ = PFLayer::NONE;

    /// rechit energy
    float energy_ = 0;

    /// time
    float time_ = -1;

    /// depth
    int depth_ = 0;

    /// indices to existing neighbours (1 common side)
    std::vector<unsigned int> neighbours_;
    std::vector<unsigned short> neighbourInfos_;

    //Caching the neighbours4/8 per request of Lindsey
    unsigned int neighbours4_ = 0;
    unsigned int neighbours8_ = 0;

    // Detector-dependent hit status flag
    uint32_t flags_ = 0;
  };

}  // namespace reco
std::ostream& operator<<(std::ostream& out, const reco::PFRecHit& hit);

#endif
