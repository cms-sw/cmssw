#ifndef RecoPPS_Local_interface_RPixClusterToHit_h
#define RecoPPS_Local_interface_RPixClusterToHit_h

/**********************************************************************
 *
 * Author: F.Ferro - INFN Genova
 *
 **********************************************************************/

#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RPixClusterToHit {
public:
  RPixClusterToHit(edm::ParameterSet const &conf);

  void buildHits(unsigned int detId,
                 const std::vector<CTPPSPixelCluster> &clusters,
                 std::vector<CTPPSPixelRecHit> &hits,
                 const PPSPixelTopology &ppt) const;

private:
  void makeHit(CTPPSPixelCluster cluster, std::vector<CTPPSPixelRecHit> &hits, PPSPixelTopology const &ppt) const;

  const int verbosity_;
};

#endif  // RecoPPS_Local_interface_RPixClusterToHit_h
