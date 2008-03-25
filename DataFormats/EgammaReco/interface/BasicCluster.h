#ifndef EgammaReco_BasicCluster_h
#define EgammaReco_BasicCluster_h
/** \class reco::BasicCluster BasicCluster.h DataFormats/EgammaReco/interface/BasicCluster.h
 *  
 * A BasicCluster reconstructed in the Electromagnetic Calorimeter
 * contains references to constituent RecHits
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: BasicCluster.h,v 1.8 2007/02/06 23:51:27 futyand Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/EcalCluster.h"

#include "Rtypes.h"

namespace reco {
  
  enum AlgoId { island = 0, hybrid = 1 };

  class BasicCluster : public EcalCluster {
  public:

    typedef math::XYZPoint Point;

    /// default constructor
    BasicCluster() : EcalCluster(0., Point(0.,0.,0.)), chi2_(-1.) { }

    BasicCluster( const double energy, const Point& position, const double chi2, const std::vector<DetId> usedHits, AlgoId algoID = hybrid);

    /// DetIds of component RecHits
    virtual std::vector<DetId> getHitsByDetId() const { return usedHits_; }

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
