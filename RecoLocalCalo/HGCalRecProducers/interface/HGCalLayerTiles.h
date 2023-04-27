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
    int getDim2Bin(float dim2) const {
      if (std::is_same_v<WRAPPER, NoPhiWrapper>) {
          return getDim1Bin(dim2);
      }
      else{
        auto normPhi = normalizedPhi(dim2);
        constexpr float r = T::nRowsPhi * M_1_PI * 0.5f;
        int phiBin = (normPhi + M_PI) * r;
        return phiBin;
      }
    }

    int mPiPhiBin = get2Bin(-M_PI);
    int pPiPhiBin = get2Bin(M_PI);


    inline float distance2(float dim1Cell1, float dim2Cell1, float dim1Cell2, float dim2Cell2) const {  // distance squared
        float d1 = dim1Cell1 - dim1Cell2;
        float d2 = dim2Cell1 - dim2Cell2;
        if(std::is_same_v<WRAPPER, PhiWrapper>){
          d2 = reco::deltaPhi(dim2Cell1, dim2Cell2);
        }
        return (d1 * d1 + d2 * d2);
    }
    int getGlobalBin(float dim1, float dim2) const { return getDim1Bin(dim1) + getDim2Bin(dim2) * T::nColumns; }

    int getGlobalBinByBin(int dim1Bin, int dim2Bin) const { return dim1Bin + dim2Bin * T::nColumns; }

  std::array<int, 4> searchBox(float dim1Min, float dim1Max, float dim2Min, float dim2Max) const {
    if (std::is_same_v<WRAPPER, NoPhiWrapper>) {
      int dim1BinMin = getDim1Bin(dim1Min);
      int dim1BinMax = getDim1Bin(dim1Max);
      int dim2BinMin = getDim2Bin(dim2Min);
      int dim2BinMax = getDim2Bin(dim2Max);
      return std::array<int, 4>({{dim1BinMin, dim1BinMax, dim2BinMin, dim2BinMax}});
    }
    else{
      if (dim1Max - dim1Min < 0) {
        return std::array<int, 4>({{0, 0, 0, 0}});
      }
      int dim1BinMin = getDim1Bin(dim1Min);
      int dim1BinMax = getDim1Bin(dim1Max);
      int dim2BinMin = getDim2Bin(dim2Min);
      int dim2BinMax = getDim2Bin(dim2Max);
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
  }

    void clear() {
        for (auto& t : tiles_)
            t.clear();
    }

    const std::vector<int>& operator[](int globalBinId) const { return tiles_[globalBinId]; }

private:
    std::array<std::vector<int>, T::nTiles> tiles_;
};

using HGCalSiliconLayerTiles = HGCalLayerTilesT<HGCalSiliconTilesConstants, NoPhiWrapper>;
using HGCalScintillatorLayerTiles = HGCalLayerTilesT<HGCalScintillatorTilesConstants, PhiWrapper>;
using HFNoseLayerTiles = HGCalLayerTilesT<HFNoseTilesConstants, NoPhiWrapper>;
#endif
