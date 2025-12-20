#ifndef DataFormats_VertexSoA_ZVertexHost_H
#define DataFormats_VertexSoA_ZVertexHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: why are the Collections not in the reco namespace like for Tracks?
using ZVertexHost = PortableHostCollection<reco::ZVertexBlocks>;

#endif  // DataFormats_VertexSoA_ZVertexHost_H
