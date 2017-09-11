/**********************************************************************
 *
 * Author: F.Ferro - INFN Genova
 *
 **********************************************************************/
#ifndef RecoCTPPS_PixelLocal_RPixClusterToHit_H
#define RecoCTPPS_PixelLocal_RPixClusterToHit_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "Geometry/VeryForwardGeometry/interface/CTPPSPixelSimTopology.h"

class RPixClusterToHit{

public:

  RPixClusterToHit(edm::ParameterSet const& conf);

  void buildHits(unsigned int detId, const std::vector<CTPPSPixelCluster> &clusters, std::vector<CTPPSPixelRecHit> &hits);
  void make_hit(CTPPSPixelCluster aCluster,  std::vector<CTPPSPixelRecHit> &hits );
  ~RPixClusterToHit();

private:

  int verbosity_;

};

#endif
