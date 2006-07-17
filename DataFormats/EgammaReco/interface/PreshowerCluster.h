#ifndef DataFormats_EgammaReco_PreshowerCluster_h
#define DataFormats_EgammaReco_PreshowerCluster_h
//
// $Id: $
//
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <cmath>

namespace reco {

  // should we inherit it from EcalCluster ???
  class PreshowerCluster { // : public EcalCluster {
  public:

    typedef math::XYZPoint Point;

    // default constructor
    PreshowerCluster();

    ~PreshowerCluster();

    // Constructor from EcalRecHits
    PreshowerCluster(const Point& position, const EcalRecHitCollection & rhits_, int layer_);

    // Constructor from cluster
    PreshowerCluster(const PreshowerCluster&);

    Point Position() const {
      return Point(radius*cos(phi)*sin(theta),
                   radius*sin(phi)*sin(theta),
                   radius*cos(theta));
    }

    double Ex() const {
      return energy*cos(phi)*sin(theta);
    }

    double Ey() const {
      return energy*sin(phi)*sin(theta);
    }

    double Ez() const {
      return energy*cos(theta);
    }

    double Et() const {
      return energy*sin(theta);
    }

// Methods that return information about the cluster
    double Energy() const {return energy;}
    double Radius() const {return radius;}
    double Theta() const {return theta;}
    double Eta() const {return eta;}
    double Phi() const {return phi;}
    int Nhits() const {return rhits.size();}

    EcalRecHitCollection::const_iterator RHBegin() const {
      return rhits.begin();
    }

    EcalRecHitCollection::const_iterator RHEnd() const {
      return rhits.end();
    }

    int Plane() {
      return layer;
    }

    static const char * name() {return "PreshowerCluster";}

    // Comparisons
    bool operator==(const PreshowerCluster&) const;
    bool operator<(const PreshowerCluster&) const;

    reco::BasicCluster * getBCPtr() {return bc_ptr;}

  private:

    double energy;
    double euncorrected;
    double et;

    double radius;
    double theta;
    double eta;
    double phi;
    int layer;

    reco::BasicCluster *bc_ptr;
    EcalRecHitCollection rhits;
  };
}
#endif
