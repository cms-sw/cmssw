// Authors: todo

#ifndef RecoLocalCalo_HGCalRecProducers_HGCalTilesWrapper_h
#define RecoLocalCalo_HGCalRecProducers_HGCalTilesWrapper_h

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTiles.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

template <typename T, typename WRAPPER >
class HGCalLayerTilesT;

template <typename T>
struct HGCalSiliconWrapper{
    static std::array<int, 4> searchBox(const HGCalLayerTilesT<T, HGCalSiliconWrapper> &tile, float dim1Min, float dim1Max, float dim2Min, float dim2Max) {
        int dim1BinMin = tile.getBin(dim1Min);
        int dim1BinMax = tile.getBin(dim1Max);
        int dim2BinMin = tile.get2Bin(dim2Min);
        int dim2BinMax = tile.get2Bin(dim2Max);
        return std::array<int, 4>({{dim1BinMin, dim1BinMax, dim2BinMin, dim2BinMax}});

    }
    static int get2Bin(const HGCalLayerTilesT<T, HGCalSiliconWrapper> &tile, float dim2){
        return tile.getBin(dim2);
    }
};

template <typename T>
struct HGCalScintillatorWrapper{
    static std::array<int, 4> searchBox(const HGCalLayerTilesT<T, HGCalScintillatorWrapper> &tile, float dim1Min, float dim1Max, float dim2Min, float dim2Max){
        if (dim1Max - dim1Min < 0) {
            return std::array<int, 4>({{0, 0, 0, 0}});
        }
        int dim1BinMin = tile.getBin(dim1Min);
        int dim1BinMax = tile.getBin(dim1Max);
        int dim2BinMin = tile.get2Bin(dim2Min);
        int dim2BinMax = tile.get2Bin(dim2Max);
        // If the search window cross the phi-bin boundary, add T::nPhiBins to the
        // MAx value. This guarantees that the caller can perform a valid doule
        // loop on eta and phi. It is the caller responsibility to perform a module
        // operation on the phiBin values returned by this function, to explore the
        // correct bins.
        if (dim2BinMax < dim2BinMin) {
            dim2BinMax += T::nRowsPhi;
        }
        return std::array<int, 4>({{dim1BinMin, dim1BinMax, dim2BinMin, dim2BinMax}});

    }
    static int get2Bin(const HGCalLayerTilesT<T, HGCalScintillatorWrapper> &tile, float dim2){
        auto normPhi = normalizedPhi(dim2);
        constexpr float r = T::nRowsPhi * M_1_PI * 0.5f;
        int phiBin = (normPhi + M_PI) * r;
        return phiBin;
    }
};

#endif