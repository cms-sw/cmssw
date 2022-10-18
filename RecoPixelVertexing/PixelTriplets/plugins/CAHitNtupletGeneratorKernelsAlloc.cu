#include "CAHitNtupletGeneratorKernelsAlloc.cc"

template class CAHitNtupletGeneratorKernelsBaseT<cms::cudacompat::GPUTraits, pixelTopology::Phase1>;
template class CAHitNtupletGeneratorKernelsBaseT<cms::cudacompat::GPUTraits, pixelTopology::Phase2>;

template class CAHitNtupletGeneratorKernelsGPUT<cms::cudacompat::GPUTraits, pixelTopology::Phase1>;
template class CAHitNtupletGeneratorKernelsGPUT<cms::cudacompat::GPUTraits, pixelTopology::Phase2>;
