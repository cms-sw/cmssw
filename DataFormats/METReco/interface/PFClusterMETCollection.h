#ifndef METRECO_PFClusterMETCOLLECTION_H
#define METRECO_PFClusterMETCOLLECTION_H

/** \class PFClusterMETCollection
 *
 * \short Collection of PFCluster MET
 *
 * PFClusterMETCollection is a collection of PFClusterMET objects
 *
 * \author Salvatore Rappoccio, JHU
 *
 * \version   1st Version Dec, 2010
 *
 ************************************************************/

#include <vector>
#include "DataFormats/METReco/interface/PFClusterMETFwd.h" 

namespace reco
{
  typedef std::vector<reco::PFClusterMET> PFClusterMETCollection;
}  
#endif // METRECO_PFClusterMETCOLLECTION_H
