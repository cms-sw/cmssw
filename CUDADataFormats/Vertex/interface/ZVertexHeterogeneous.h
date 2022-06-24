#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "CUDADataFormats/Vertex/interface/ZVertexSoA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memoryPool.h"

using ZVertexHeterogeneous = memoryPool::Buffer<ZVertexSoA>;
#ifndef __CUDACC__
#include "CUDADataFormats/Common/interface/Product.h"
using ZVertexCUDAProduct = cms::cuda::Product<ZVertexHeterogeneous>;
#endif

#endif
