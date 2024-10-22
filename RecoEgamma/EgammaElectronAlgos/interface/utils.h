#ifndef RecoEgamma_EgammaElectronAlgos_utils_h
#define RecoEgamma_EgammaElectronAlgos_utils_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <unordered_map>
#include <cstring>

struct HashIntGlobalPointPair {
  inline std::size_t operator()(std::pair<int, GlobalPoint> const& g) const {
    auto h1 = std::hash<unsigned long long>()((unsigned long long)g.first);
    unsigned long long k;
    memcpy(&k, &g.second, sizeof(k));
    auto h2 = std::hash<unsigned long long>()(k);
    return h1 ^ (h2 << 1);
  }
};

template <class T>
using IntGlobalPointPairUnorderedMap = std::unordered_map<std::pair<int, GlobalPoint>, T, HashIntGlobalPointPair>;

#endif
