#ifndef EgammaReco_BasicCluster_h
#define EgammaReco_BasicCluster_h
/** \class reco::BasicCluster BasicCluster.h DataFormats/EgammaReco/interface/BasicCluster.h
 *  
 * A BasicCluster reconstructed in the Electromagnetic Calorimeter
 * contains references to constituent RecHits
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: BasicCluster.h,v 1.12 2008/03/02 12:32:42 kkaadze Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "Rtypes.h" 
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include <vector>

namespace reco {
  
  enum AlgoId { island = 0, hybrid = 1, fixedMatrix = 2, dynamicHybrid = 3 };

  class BasicCluster : public CaloCluster {
  public:

    typedef math::XYZPoint Point;

    /// default constructor
    BasicCluster() : CaloCluster(0., Point(0.,0.,0.)), chi2_(-1.) { }

    BasicCluster( const double energy, const Point& position, const double chi2, const std::vector<DetId> usedHits, AlgoId algoID = hybrid);

    /// DetIds of component RecHits
    std::vector<DetId> getHitsByDetId() const { return usedHits_; }

    /// chi-squared
    double chi2() const { return chi2_; }

    /// identifier of the algorithm
    AlgoId algo() const { return algoId_; }

    /// this method is needed to sort the BasicClusters by energy
    bool operator<(const reco::BasicCluster &otherCluster) const;
    bool operator==(const BasicCluster& rhs) const;

  private:

    /// chi-squared
    Double32_t chi2_;

    /// 0 in case of island algorithm, 1 in case of hybrid
    AlgoId algoId_;

    /// used hits by detId
    std::vector<DetId> usedHits_;

  };


}

#endif
