#ifndef EGAMMATOOLS_EcalClusterLocal_h
#define EGAMMATOOLS_EcalClusterLocal_h

/** \class EcalClusterLocal
  *  Function to compute local coordinates of Ecal clusters
  *  (adapted from RecoEcal/EgammaCoreTools/plugins/EcalClusterLocal)
  *  \author Josh Bendavid, MIT, 2011
  */

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

class CaloGeometry;

namespace egammaTools {

  void localEcalClusterCoordsEB(const reco::CaloCluster &bclus,
                                const CaloGeometry &geom,
                                float &etacry,
                                float &phicry,
                                int &ieta,
                                int &iphi,
                                float &thetatilt,
                                float &phitilt);
  void localEcalClusterCoordsEE(const reco::CaloCluster &bclus,
                                const CaloGeometry &geom,
                                float &xcry,
                                float &ycry,
                                int &ix,
                                int &iy,
                                float &thetatilt,
                                float &phitilt);

};  // namespace egammaTools

#endif
