#ifndef EgammaReco_PreShowerCluster_h
#define EgammaReco_PreShowerCluster_h
/** \class reco::PreShowerCluster PreShowerCluster.h DataFormats/EgammaReco/interface/PreShowerCluster.h
 *  
 * A cluster reconstructed in ECAL pre shower
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: PreShowerCluster.h,v 1.1 2006/04/09 15:40:40 rahatlou Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/PreShowerClusterFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {
  class PreShowerCluster {
  public:
    /// default constructor
    PreShowerCluster();
  private:
  };

}

#endif
