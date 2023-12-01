/* \class PixelOccupancyFilter
 *
 * Filters events if at least N pixel modules have M clusters, where A <=  M <= B 
 *
 * \author: Marco Musich
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/UtilAlgos/interface/DetSetCounterSelector.h"

typedef ObjectCountFilter<SiPixelClusterCollectionNew, DetSetCounterSelector>::type PixelOccupancyFilter;

DEFINE_FWK_MODULE(PixelOccupancyFilter);
