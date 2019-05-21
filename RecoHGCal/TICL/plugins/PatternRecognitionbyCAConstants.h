#ifndef __RecoHGCal_TICL_PRbyCAConstants_H__
#define __RecoHGCal_TICL_PRbyCAConstants_H__

namespace ticl::patternbyca {
    constexpr int nEtaBins = 34;
    constexpr int nPhiBins = 126;
    constexpr int nLayers = 104;
    typedef std::array<std::array<std::vector<unsigned int>, nEtaBins*nPhiBins>, nLayers>
      Tile;
}


#endif
