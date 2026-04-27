#ifndef DataFormats_VertexSoA_interface_VertexSoA_h
#define DataFormats_VertexSoA_interface_VertexSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace {
  constexpr int maxTracksPerVertex = 1024;
}
using VertexToTrack = Eigen::Vector<float, maxTracksPerVertex>;
using VertexToTrackInt = Eigen::Vector<int, maxTracksPerVertex>;

GENERATE_SOA_LAYOUT(
    VertexSoALayout,
    SOA_COLUMN(float, x),
    SOA_COLUMN(float, y),
    SOA_COLUMN(float, z),
    SOA_COLUMN(float, t),

    SOA_COLUMN(float, errx),
    SOA_COLUMN(float, erry),
    SOA_COLUMN(float, errz),
    SOA_COLUMN(float, errt),

    SOA_COLUMN(float, chi2),
    SOA_COLUMN(float, ndof),
    SOA_COLUMN(int, ntracks),
    SOA_COLUMN(float, rho),

    SOA_COLUMN(float, aux1),
    SOA_COLUMN(float, aux2),

    SOA_EIGEN_COLUMN(VertexToTrackInt, track_id),
    SOA_EIGEN_COLUMN(VertexToTrack, track_weight),

    SOA_COLUMN(bool, isGood),
    SOA_COLUMN(int, order),

    SOA_COLUMN(float, sw),
    SOA_COLUMN(float, swE),

    // Use entries for blocks
    // When running in paralell, i.e. DA in blocks, the entries in the column can be used to save the number of vertex on each block during clustering
    SOA_COLUMN(int32_t, nV))

using VertexSoA = VertexSoALayout<>;

#endif  // DataFormats_VertexSoA_interface_VertexSoA_h
