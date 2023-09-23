/* \class PixelCountFilter
 *
 * Filters events if at least N pixel clusters
 *
 * \author: Vincenzo Chiochia, Uni-ZH
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "CommonTools/UtilAlgos/interface/CollectionInCollectionFilter.h"

typedef CollectionInCollectionFilter<SiPixelClusterCollectionNew>::type PixelCountFilter;

DEFINE_FWK_MODULE(PixelCountFilter);
