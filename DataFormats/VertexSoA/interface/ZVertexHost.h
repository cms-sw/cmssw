#ifndef DataFormats_VertexSoA_ZVertexHost_H
#define DataFormats_VertexSoA_ZVertexHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

using ZVertexHost = PortableHostCollection2<reco::ZVertexSoA, reco::ZVertexTracksSoA>;

#endif  // DataFormats_VertexSoA_ZVertexHost_H
