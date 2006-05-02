#ifndef PreshowerCluster_h
#define PreshowerCluster_h

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include <cmath>
#include <vector>

namespace reco {
  
  class PreshowerCluster {
  
  public:

    typedef math::XYZPoint Point;
    
    // default constructor
    PreshowerCluster();

    

    virtual ~PreshowerCluster();

    // Constructor from EcalRecHits
    PreshowerCluster(const std::vector<EcalRecHit*> &rhits,
		     int layer_,
		     const CaloSubdetectorGeometry *geometry_p,
		     const CaloSubdetectorTopology *topology_p);


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

    double Energy() const;
    double EnergyUncorrected() const;
    double Radius() const;
    double Theta() const;
    double Eta() const;
    double Phi() const;
    int Nhits() const;

    std::vector<EcalRecHit*>::const_iterator RHBegin() const {
      return rhits.begin();
    }
      
    std::vector<EcalRecHit*>::const_iterator RHEnd() const {
      return rhits.end();
    }

    int Plane() {
      return layer_;
    }

    // Cluster correction
    void Correct();

    static const char * name() {return "PreshowerCluster";}

    // Comparisons
    int operator==(const PreshowerCluster&) const;
    int operator<(const PreshowerCluster&) const;

    reco::BasicCluster * getBCPtr() {return bc_ptr;}

  protected:



    double energy;
    double et;

    double radius;
    double theta;
    double eta;
    double phi;

    double x;
    double y;
    double z;

    double euncorrected;
 
    int nhits;

    virtual void init();
    virtual void init(const PreshowerCluster &);
    
    CaloSubdetectorGeometry geometry;
 
    CaloSubdetectorTopology topology;

    reco::BasicCluster *bc_ptr;

  private:
    
    int layer_;

    std::vector<EcalRecHit*> rhits;
  };

}

#endif
