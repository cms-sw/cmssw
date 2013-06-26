#ifndef CastorReco_CastorEgamma_h
#define CastorReco_CastorEgamma_h
/** \class reco::CastorEgamma CastorEgamma.h DataFormats/CastorReco/CastorEgamma.h
 *  
 * Class for Castor electrons/photons
 *
 * \author Hans Van Haevermaet, University of Antwerp
 *
 * \version $Id: CastorEgamma.h,v 1.6 2010/07/03 19:13:10 hvanhaev Exp $
 *
 */
#include <vector>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"

namespace reco {

  class CastorEgamma : public CastorCluster {
  public:

    /// default constructor. Sets energy to zero
    CastorEgamma() : energycal_(0.) { }

    /// constructor from values
    CastorEgamma(const double energycal, const CastorClusterRef& usedCluster);

    /// destructor
    virtual ~CastorEgamma();

    /// Egamma energy
    double energy() const { return (*usedCluster_).energy(); }

    /// Egamma energycal
    double energycal() const { return energycal_; }

    /// Egamma centroid position
    ROOT::Math::XYZPoint position() const { return (*usedCluster_).position(); }
    
    /// vector of used Clusters
    CastorClusterRef getUsedCluster() const { return usedCluster_; }

    /// comparison >= operator
    bool operator >=(const CastorEgamma& rhs) const { return (energycal_>=rhs.energycal_); }

    /// comparison > operator
    bool operator > (const CastorEgamma& rhs) const { return (energycal_> rhs.energycal_); }

    /// comparison <= operator
    bool operator <=(const CastorEgamma& rhs) const { return (energycal_<=rhs.energycal_); }

    /// comparison <= operator
    bool operator < (const CastorEgamma& rhs) const { return (energycal_< rhs.energycal_); }

    /// Egamma em energy
    double emEnergy() const { return (*usedCluster_).emEnergy(); }

    /// Egamma had energy
    double hadEnergy() const { return (*usedCluster_).hadEnergy(); }

    /// Egamma em/tot ratio
    double fem() const { return (*usedCluster_).fem(); }

    /// Egamma width in phi
    double width() const { return (*usedCluster_).width(); }

    /// Egamma depth in z
    double depth() const { return (*usedCluster_).depth(); }

    /// Egamma hotcell/tot ratio
    double fhot() const { return (*usedCluster_).fhot(); }

    /// Egamma sigma z
    double sigmaz() const { return (*usedCluster_).sigmaz(); }

    /// pseudorapidity of Egamma centroid
    double eta() const { return (*usedCluster_).eta(); }

    /// azimuthal angle of Egamma centroid
    double phi() const { return (*usedCluster_).phi(); }

    /// x of Egamma centroid
    double x() const { return (*usedCluster_).x(); }

    /// y of Egamma centroid
    double y() const { return (*usedCluster_).y(); }

    /// rho of Egamma centroid
    double rho() const { return (*usedCluster_).rho(); }

  private:

    /// Egamma energycal
    double energycal_;

    /// used CastorClusters
    CastorClusterRef usedCluster_;
  };
  
  // define CastorEgammaCollection
  typedef std::vector<CastorEgamma> CastorEgammaCollection;

}

#endif
