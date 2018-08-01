//
// Author: Felice Pantaleo, CERN
//

#ifndef RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h
#define RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h

#include <vector>

struct GPULayerHits
{
        unsigned int layerId;
        size_t size;
        float * x;
        float * y;
        float * z;
};

struct HostLayerHits
{
        unsigned int layerId;
        size_t size;
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> z;
};

struct GPULayerDoublets
{
        size_t size;
        unsigned int innerLayerId;
        unsigned int outerLayerId;
        unsigned int * indices;
};

struct HostLayerDoublets
{
        size_t size;
        unsigned int innerLayerId;
        unsigned int outerLayerId;
        std::vector<unsigned int> indices;
};

#endif
