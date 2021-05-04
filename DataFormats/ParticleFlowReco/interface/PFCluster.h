#ifndef DataFormats_ParticleFlowReco_PFCluster_h
#define DataFormats_ParticleFlowReco_PFCluster_h

#include <iostream>
#include <vector>
#include <algorithm>

#include <Rtypes.h>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "Math/GenVector/PositionVector3D.h"

class PFClusterAlgo;

namespace reco {

  /**\class PFCluster
     \brief Particle flow cluster, see clustering algorithm in PFClusterAlgo
     
     A particle flow cluster is defined by its energy and position, which are 
     calculated from a vector of PFRecHitFraction. This calculation is 
     performed in PFClusterAlgo.

     \todo Clean up this class to a common base (talk to Paolo Meridiani)
     the extra internal stuff (like the vector of PFRecHitFraction's)
     could be moved to a PFClusterExtra.
     
     \todo Now that PFRecHitFraction's hold a reference to the PFRecHit's, 
     put back the calculation of energy and position to PFCluster. 


     \todo Add an operator+=

     \author Colin Bernet
     \date   July 2006
  */
  class PFCluster : public CaloCluster {
  public:
    typedef std::vector<std::pair<CaloClusterPtr::key_type, edm::Ptr<PFCluster>>> EEtoPSAssociation;
    // Next typedef uses double in ROOT 6 rather than Double32_t due to a bug in ROOT 5,
    // which otherwise would make ROOT5 files unreadable in ROOT6.  This does not increase
    // the size on disk, because due to the bug, double was actually stored on disk in ROOT 5.
    typedef ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double>> REPPoint;

    PFCluster() : CaloCluster(CaloCluster::particleFlow), time_(-99.0), layer_(PFLayer::NONE) {}

    /// constructor
    PFCluster(PFLayer::Layer layer, double energy, double x, double y, double z);

    /// resets clusters parameters
    void reset();

    /// reset only hits and fractions
    void resetHitsAndFractions();

    /// add a given fraction of the rechit
    void addRecHitFraction(const reco::PFRecHitFraction& frac);

    /// vector of rechit fractions
    const std::vector<reco::PFRecHitFraction>& recHitFractions() const { return rechits_; }

    /// set layer
    void setLayer(PFLayer::Layer layer);

    /// cluster layer, see PFLayer.h in this directory
    PFLayer::Layer layer() const;

    /// cluster energy
    double energy() const { return energy_; }

    /// \return cluster time
    float time() const { return time_; }
    /// \return the timing uncertainty
    float timeError() const { return timeError_; }

    /// cluster depth
    double depth() const { return depth_; }

    void setTime(float time, float timeError = 0) {
      time_ = time;
      timeError_ = timeError;
    }
    void setTimeError(float timeError) { timeError_ = timeError; }
    void setDepth(double depth) { depth_ = depth; }

    /// cluster position: rho, eta, phi
    const REPPoint& positionREP() const { return posrep_; }

    /// computes posrep_ once and for all
    void calculatePositionREP() { posrep_.SetCoordinates(position_.Rho(), position_.Eta(), position_.Phi()); }

    /// \todo move to PFClusterTools
    static double getDepthCorrection(double energy, bool isBelowPS = false, bool isHadron = false);

    PFCluster& operator=(const PFCluster&);

    /// some classes to make this fit into a template footprint
    /// for RecoPFClusterRefCandidate so we can make jets and MET
    /// out of PFClusters.

    /// dummy charge
    double charge() const { return 0; }

    /// transverse momentum, massless approximation
    double pt() const { return (energy() * sin(position_.theta())); }

    /// angle
    double theta() const { return position_.theta(); }

    /// dummy vertex access
    math::XYZPoint const& vertex() const { return dummyVtx_; }
    double vx() const { return vertex().x(); }
    double vy() const { return vertex().y(); }
    double vz() const { return vertex().z(); }

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    template <typename pruner>
    void pruneUsing(pruner prune) {
      // remove_if+erase algo applied to both vectors...
      auto iter = std::find_if_not(rechits_.begin(), rechits_.end(), prune);
      if (iter == rechits_.end())
        return;
      auto first = iter - rechits_.begin();
      for (auto i = first; ++i < int(rechits_.size());) {
        if (prune(rechits_[i])) {
          rechits_[first] = std::move(rechits_[i]);
          hitsAndFractions_[first] = std::move(hitsAndFractions_[i]);
          ++first;
        }
      }
      rechits_.erase(rechits_.begin() + first, rechits_.end());
      hitsAndFractions_.erase(hitsAndFractions_.begin() + first, hitsAndFractions_.end());
    }
#endif

  private:
    /// vector of rechit fractions (transient)
    std::vector<reco::PFRecHitFraction> rechits_;

    /// cluster position: rho, eta, phi (transient)
    REPPoint posrep_;

    /// Michalis: add timing and depth information
    float time_, timeError_;
    double depth_;

    /// transient layer
    PFLayer::Layer layer_;

    /// depth corrections
    static const constexpr double depthCorA_ = 0.89;
    static const constexpr double depthCorB_ = 7.3;
    static const constexpr double depthCorAp_ = 0.89;
    static const constexpr double depthCorBp_ = 4.0;

    static const math::XYZPoint dummyVtx_;
  };

  std::ostream& operator<<(std::ostream& out, const PFCluster& cluster);

}  // namespace reco

#endif  // DataFormats_ParticleFlowReco_PFCluster_h
