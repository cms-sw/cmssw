// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019

#ifndef RecoLocalCalo_HGCalRecProducers_HGCalLayerTiles_h
#define RecoLocalCalo_HGCalRecProducers_HGCalLayerTiles_h

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HFNoseTilesConstants.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesWrapper.h"

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>

template <typename T, typename WRAPPER >
class HGCalLayerTilesT {
public:
    typedef T type;
    void fill(const std::vector<float>& dim1,
              const std::vector<float>& dim2) {
        auto cellsSize = dim1.size();
        for (unsigned int i = 0; i < cellsSize; ++i) {
            auto idx = getGlobalBin(dim1[i], dim2[i]);
            tiles_[idx].push_back(i);
        }
    }
    /**
    * this is for x and eta
    */
    int getBin(float x) const {
        constexpr float xRange = T::maxDim - T::minDim;
        static_assert(xRange >= 0.);
        constexpr float r = T::nColumns / xRange;
        int xBin = (x - T::minDim) * r;
        xBin = std::clamp(xBin, 0, T::nColumns - 1);
        return xBin;
    }

    /**
    * this is for phi and y
    */
    int get2Bin(float dim2) const{
        return WRAPPER::get2Bin(*this, dim2);
    }

    int mPiPhiBin = get2Bin(-M_PI);
    int pPiPhiBin = get2Bin(M_PI);

    int getGlobalBin(float dim1, float dim2) const { return getBin(dim1) + get2Bin(dim2) * T::nColumns; }

    int getGlobalBinByBin(int dim1Bin, int dim2Bin) const { return dim1Bin + dim2Bin * T::nColumns; }

    std::array<int, 4> searchBox(float dim1Min, float dim1Max, float dim2Min, float dim2Max) const {
        return WRAPPER::searchBox(*this, dim1Min, dim1Max, dim2Min, dim2Max);
    }

    void clear() {
        for (auto& t : tiles_)
            t.clear();
    }

    const std::vector<int>& operator[](int globalBinId) const { return tiles_[globalBinId]; }

private:
    std::array<std::vector<int>, T::nTiles> tiles_;
};

using HGCalSiliconLayerTiles = HGCalLayerTilesT<HGCalSiliconTilesConstants, HGCalSiliconWrapper<HGCalSiliconTilesConstants>>;
using HGCalScintillatorLayerTiles = HGCalLayerTilesT<HGCalScintillatorTilesConstants, HGCalScintillatorWrapper<HGCalScintillatorTilesConstants>>;
using HFNoseLayerTiles = HGCalLayerTilesT<HFNoseTilesConstants, HGCalScintillatorWrapper<HFNoseTilesConstants>>;
#endif
