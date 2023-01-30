#ifndef RecoTracker_DisplacedRegionalTracking_DisplacedVertexCluster_h
#define RecoTracker_DisplacedRegionalTracking_DisplacedVertexCluster_h

#include <cmath>
#include <limits>
#include <list>
#include <utility>

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Math/interface/Vector3D.h"

class DisplacedVertexCluster;
typedef std::list<DisplacedVertexCluster>::iterator DisplacedVertexClusterItr;

class DisplacedVertexCluster {
public:
  static constexpr double kInvalidDouble = std::numeric_limits<double>::quiet_NaN();

  DisplacedVertexCluster()
      : valid_(false), rParam2_(kInvalidDouble), sumOfCenters_(0.0, 0.0, 0.0), centerOfMass_(0.0, 0.0, 0.0) {}

  DisplacedVertexCluster(const edm::View<reco::VertexCompositeCandidate> &, const unsigned, const double);

  ~DisplacedVertexCluster() { constituents_.clear(); }

  bool valid() const { return valid_; }
  double rParam2() const { return rParam2_; }
  double rParam() const { return sqrt(rParam2()); }
  const std::vector<const reco::VertexCompositeCandidate *> &constituents() const { return constituents_; }
  const reco::VertexCompositeCandidate *constituent(const unsigned i) const { return constituents_.at(i); }
  unsigned nConstituents() const { return constituents_.size(); }
  const math::XYZVector &sumOfCenters() const { return sumOfCenters_; }
  const math::XYZVector &centerOfMass() const { return centerOfMass_; }

  double vx() const { return centerOfMass().x(); }
  double vy() const { return centerOfMass().y(); }
  double vz() const { return centerOfMass().z(); }

  void merge(const DisplacedVertexCluster &other);
  void setInvalid() { valid_ = false; }

  // struct representing the distance between two DisplacedVertexCluster objects
  struct Distance {
  public:
    Distance(DisplacedVertexClusterItr entity0, DisplacedVertexClusterItr entity1) : entities_(entity0, entity1) {}
    double distance2() const;
    double distance() const { return sqrt(distance2()); }
    std::pair<DisplacedVertexClusterItr, DisplacedVertexClusterItr> &entities() { return entities_; }
    const std::pair<DisplacedVertexClusterItr, DisplacedVertexClusterItr> &entities() const { return entities_; }

  private:
    std::pair<DisplacedVertexClusterItr, DisplacedVertexClusterItr> entities_;
  };

  typedef std::list<Distance>::iterator DistanceItr;

private:
  bool valid_;
  double rParam2_;
  std::vector<const reco::VertexCompositeCandidate *> constituents_;
  math::XYZVector sumOfCenters_;
  math::XYZVector centerOfMass_;
};

#endif
