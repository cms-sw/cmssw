/**********************************************************************
 *
 * Author: F.Ferro - INFN Genova
 *
 **********************************************************************/
#ifndef RecoPPS_Local_RPixClusterToHit_H
#define RecoPPS_Local_RPixClusterToHit_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"

class RPixClusterToHit {
public:
  RPixClusterToHit(edm::ParameterSet const &conf);

  void buildHits(unsigned int detId,
                 const std::vector<CTPPSPixelCluster> &clusters,
                 std::vector<CTPPSPixelRecHit> &hits,
                 const PPSPixelTopology &ppt);
  void make_hit(CTPPSPixelCluster aCluster, std::vector<CTPPSPixelRecHit> &hits, const PPSPixelTopology &ppt);
  ~RPixClusterToHit();

private:
  int verbosity_;
};

#endif
