#ifndef DataFormats_EgammaReco_PreshowerClusterShape_h
#define DataFormats_EgammaReco_PreshowerClusterShape_h
/*
 * PreshowerShape cluster class
 *
 * \author Aris Kyriakis (NCSR "Demokritos")
 */
//
#include "DataFormats/EgammaReco/interface/PreshowerClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

namespace reco {

  class PreshowerClusterShape {
  public:

    /// default constructor
    PreshowerClusterShape() { };

    virtual ~PreshowerClusterShape();

    /// constructor from strip energies
    PreshowerClusterShape(const std::vector<float>& stripEnergies, 
                     const int plane);		    

    /// Copy contructor
    PreshowerClusterShape(const PreshowerClusterShape&);

    /// Preshower plane
    int plane() const { return plane_; }
   
    /// Associated SuperCluster;
    SuperClusterRef superCluster() const {return sc_ref_;}

    /// Energies of component strips
    virtual std::vector<float> getStripEnergies() const { return stripEnergies_; }

    void setSCRef( const SuperClusterRef & r ) { sc_ref_ = r; }

  private:

    int plane_;

    /// Associated super cluster;
    SuperClusterRef sc_ref_;

    /// used strip energies
    std::vector<float> stripEnergies_;
  };
}
#endif
