#ifndef CastorReco_CastorJet_h
#define CastorReco_CastorJet_h
/** \class reco::CastorJet CastorJet.h DataFormats/CastorReco/CastorJet.h
 *  
 * Class for Castor electrons/photons
 *
 * \author Hans Van Haevermaet, University of Antwerp
 *
 * \version $Id: CastorJet.h,v 1.6 2010/07/03 19:12:57 hvanhaev Exp $
 *
 */
#include <vector>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"

namespace reco {

  class CastorJet : public CastorCluster {
  public:

    /// default constructor. Sets energy to zero
    CastorJet() : energycal_(0.) { }

    /// constructor from values
    CastorJet(const double energycal, const CastorClusterRef& usedCluster);

    /// destructor
    virtual ~CastorJet();

    /// Jet energy
    double energy() const { return (*usedCluster_).energy(); }

    /// Jet energycal
    double energycal() const { return energycal_; }

    /// Jet centroid position
    ROOT::Math::XYZPoint position() const { return (*usedCluster_).position(); }
    
    /// vector of used Clusters
    CastorClusterRef getUsedCluster() const { return usedCluster_; }

    /// comparison >= operator
    bool operator >=(const CastorJet& rhs) const { return (energycal_>=rhs.energycal_); }

    /// comparison > operator
    bool operator > (const CastorJet& rhs) const { return (energycal_> rhs.energycal_); }

    /// comparison <= operator
    bool operator <=(const CastorJet& rhs) const { return (energycal_<=rhs.energycal_); }

    /// comparison <= operator
    bool operator < (const CastorJet& rhs) const { return (energycal_< rhs.energycal_); }

    /// Jet em energy
    double emEnergy() const { return (*usedCluster_).emEnergy(); }

    /// Jet had energy
    double hadEnergy() const { return (*usedCluster_).hadEnergy(); }

    /// Jet em/tot ratio
    double fem() const { return (*usedCluster_).fem(); }

    /// Jet width in phi
    double width() const { return (*usedCluster_).width(); }

    /// Jet depth in z
    double depth() const { return (*usedCluster_).depth(); }

    /// Jet hotcell/tot ratio
    double fhot() const { return (*usedCluster_).fhot(); }

    /// Jet sigma z
    double sigmaz() const { return (*usedCluster_).sigmaz(); }

    /// pseudorapidity of Jet centroid
    double eta() const { return (*usedCluster_).eta(); }

    /// azimuthal angle of Jet centroid
    double phi() const { return (*usedCluster_).phi(); }

    /// x of Jet centroid
    double x() const { return (*usedCluster_).x(); }

    /// y of Jet centroid
    double y() const { return (*usedCluster_).y(); }

    /// rho of Jet centroid
    double rho() const { return (*usedCluster_).rho(); }

  private:

    /// Jet energycal
    double energycal_;

    /// used CastorClusters
    CastorClusterRef usedCluster_;
  };
  
  // define CastorJetCollection
  typedef std::vector<CastorJet> CastorJetCollection;

}

#endif
