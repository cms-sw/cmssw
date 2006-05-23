#ifndef EgammaReco_BasicCluster_h
#define EgammaReco_BasicCluster_h
/** \class reco::BasicCluster BasicCluster.h DataFormats/EgammaReco/interface/BasicCluster.h
 *  
 * A BasicCluster reconstructed in the Electromagnetic Calorimeter
 * contains references to constituent RecHits
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: BasicCluster.h,v 1.6 2006/05/10 12:45:59 tsirig Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/EcalCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"

namespace reco {
  
  class EcalRecHitData {
  public:
    EcalRecHitData() { }
    EcalRecHitData( float e, float chi2, DetId id, float frac=1 ) :
      energy_(e), chi2_(chi2), detId_(id), fraction_(frac) { }
    ~EcalRecHitData() { }
    float energy() const {return energy_;}
    float chi2() const {return chi2_;}
    DetId detId() const {return detId_;}
    float fraction() const {return fraction_;}
  private:
    float energy_;
    float chi2_;
    DetId detId_;
    float fraction_;
  };

  enum AlgoId { island = 0, hybrid = 1 };

  class BasicCluster : public EcalCluster {
  public:
    /// point in the space
    typedef math::XYZPoint Point;
    /// index type used as reference to RecHit.
    //typedef unsigned short RecHitIndex;
    /// collection of (indices to) RecHit with assoxciated energy fraction
    //typedef std::vector<std::pair<RecHitIndex, float> > RecHitCollection;

    /// default constructor
    BasicCluster() : EcalCluster(0., Point(0.,0.,0.)), chi2_(-1.) { }

    /// constructor from EcalRecHits
    //BasicCluster( const std::vector<EcalRecHitData>& recHits,
    //              int superClusterId,
    //             const Point & position = Point( 0, 0, 0 ) );

    BasicCluster( const double energy, const Point& position, const double chi2, const std::vector<DetId> usedHits, AlgoId algoID = hybrid);

    virtual std::vector<DetId> getHitsByDetId() const { return usedHits_; }

    void SetClusterShapeRef(ClusterShapeRef ref){ clusref=ref;}
    /// cluster centriod position
    //const Point & position() const { return position_; }
    /// cluster energy
    //double energy() const { return energy_; }

    /// chi-squared
    double chi2() const { return chi2_; }

    /// identifier of the algorithm
    AlgoId algo() const { return algoId_; }

    /// identifier of supercluster, or -1 in case of island algorithm
    int superClusterId() const { return algoId_; }

    /// Access to ECAL RecHit information
    //std::vector<EcalRecHitData> recHits() const { return recHits_; }

    /// this method is needed to sort the BasicClusters by energy
    bool operator<(const reco::BasicCluster &otherCluster) const;
    bool operator==(const BasicCluster& rhs) const;

  private:
    /// cluster centroid position
    //Point position_;
    /// cluster energy
    //Double32_t energy_;

    /// chi-squared
    Double32_t chi2_;

    /// 0 in case of island algorithm, 1 in case of hybrid
    AlgoId algoId_;
    //Associated clustershape information;
    ClusterShapeRef clusref;
    /// ECAL RecHit information
    //std::vector<EcalRecHitData> recHits_;

   /// used hits by detId
   std::vector<DetId> usedHits_;

  };


}

#endif
