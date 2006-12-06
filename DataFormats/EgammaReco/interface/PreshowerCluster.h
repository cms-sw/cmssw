#ifndef DataFormats_EgammaReco_PreshowerCluster_h
#define DataFormats_EgammaReco_PreshowerCluster_h
/*
 * Preshower cluster class
 *
 * \authors Dmirty Bandurin (KSU), Ted Kolberg (ND)
 */
// $Id: PreshowerCluster.h,v 1.11 2006/07/21 14:02:04 rahatlou Exp $
//
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <cmath>

namespace reco {

  class PreshowerCluster : public EcalCluster {
  public:

    typedef math::XYZPoint Point;

    // default constructor
    PreshowerCluster() : EcalCluster(0., Point(0.,0.,0.)) { };

    virtual ~PreshowerCluster();

    // Constructor from EcalRecHits
    PreshowerCluster(const double E, const Point& pos, 
                     const EcalRecHitCollection & rhits, 		     		     
                     const int plane);

    // Constructor from cluster
    PreshowerCluster(const PreshowerCluster&);

    //    Point Position() const {
    //      return Point(radius_*cos(phi)*sin(theta),
    //                   radius_*sin(phi)*sin(theta),
    //                   radius_*cos(theta));
    //    }

    double ex() const {
      return energy()*cos(phi_)*sin(theta_);
    }

    double ey() const {
      return energy()*sin(phi_)*sin(theta_);
    }

    double ez() const {
      return energy()*cos(theta_);
    }

    double et() const {
      return energy()*sin(theta_);
    }

// Methods that return information about the cluster
//    double energy() const {return energy_;}
    double radius() const {return radius_;}
    double theta() const {return theta_;}
    double eta() const {return eta_;}
    double phi() const {return phi_;}
    int nhits() const {return rhits_.size();}

    EcalRecHitCollection::const_iterator firstRecHit() const {
      return rhits_.begin();
    }

    EcalRecHitCollection::const_iterator lastRecHit() const {
      return rhits_.end();
    }

    int plane() {
      return plane_;
    }

    static const char * name() {return "PreshowerCluster";}

    // Comparisons
    bool operator==(const PreshowerCluster&) const;
    bool operator<(const PreshowerCluster&) const;

    //Associated basic cluster;
    //BasicClusterRef basicCluster() const {return bc_ref_;}

    virtual std::vector<DetId> getHitsByDetId() const { return usedHits_; }

  private:

    //    double energy_;
    double et_;

    double radius_;
    double theta_;
    double eta_;
    double phi_;

    int plane_;

    //Associated basic cluster;
    //BasicClusterRef bc_ref_;    

    //Preshower cluster rec. hits
    EcalRecHitCollection rhits_;

    // used hits by detId
    std::vector<DetId> usedHits_;
  };
}
#endif
