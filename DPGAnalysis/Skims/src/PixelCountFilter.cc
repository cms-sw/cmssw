/* \class PixelCountFilter
 *
 * Filters events if at least N pixel clusters
 *
 * \author: Vincenzo Chiochia, Uni-ZH
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/UtilAlgos/interface/DetSetCounterSelector.h"

typedef ObjectCountFilter<SiPixelClusterCollectionNew, DetSetCounterSelector>::type PixelCountFilter;

DEFINE_FWK_MODULE(PixelCountFilter);
