#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "CUDADataFormats/Vertex/interface/ZVertexSoA.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;
#ifndef __CUDACC__
#include "CUDADataFormats/Common/interface/Product.h"
using ZVertexCUDAProduct = cms::cuda::Product<ZVertexHeterogeneous>;
#endif

#endif
