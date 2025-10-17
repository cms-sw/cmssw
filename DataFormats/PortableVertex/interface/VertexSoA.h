#ifndef DataFormats_PortableVertex_interface_VertexSoA_h
#define DataFormats_PortableVertex_interface_VertexSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace portablevertex {

  using VertexToTrack = Eigen::Vector<float, 1024>;
  using VertexToTrackInt = Eigen::Vector<int, 1024>;
  // SoA layout with x, y, z, id fields
  GENERATE_SOA_LAYOUT(VertexSoALayout,
                      // columns: one value per element
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
                      SOA_COLUMN(float, se),
                      SOA_COLUMN(float, swz),
                      SOA_COLUMN(float, swE),
                      SOA_COLUMN(float, exp),
                      SOA_COLUMN(float, exparg),

                      // Use entries for blocks
                      SOA_COLUMN(int32_t, nV))

  using VertexSoA = VertexSoALayout<>;

  using TrackToVertex = Eigen::Vector<float, 512>;  // 512 is the max vertex allowed
  GENERATE_SOA_LAYOUT(TrackSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(float, dxy2),
                      SOA_COLUMN(float, dxy2AtIP),
                      SOA_COLUMN(float, dz2),
                      SOA_COLUMN(float, oneoverdz2),
                      SOA_COLUMN(float, weight),
                      SOA_COLUMN(float, sum_Z),
                      SOA_COLUMN(int, kmin),
                      SOA_COLUMN(int, kmax),
                      SOA_COLUMN(bool, isGood),
                      SOA_COLUMN(int, order),
                      SOA_COLUMN(int, tt_index),

                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),

                      SOA_COLUMN(float, xAtIP),
                      SOA_COLUMN(float, yAtIP),

                      SOA_COLUMN(float, dx),
                      SOA_COLUMN(float, dy),
                      SOA_COLUMN(float, dz),

                      SOA_COLUMN(float, dxError),
                      SOA_COLUMN(float, dyError),
                      SOA_COLUMN(float, dzError),

                      SOA_COLUMN(float, px),
                      SOA_COLUMN(float, py),
                      SOA_COLUMN(float, pz),

                      SOA_COLUMN(float, aux1),
                      SOA_COLUMN(float, aux2),

                      // Track-vertex association
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_sw),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_se),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_swz),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_swE),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_exp),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_exparg),

                      // scalars: one value for the whole structure
                      SOA_SCALAR(int32_t, nT),
                      SOA_SCALAR(float, totweight))

  using TrackSoA = TrackSoALayout<>;

  GENERATE_SOA_LAYOUT(ClusterParams,
                      SOA_SCALAR(float, d0CutOff),
                      SOA_SCALAR(float, TMin),
                      SOA_SCALAR(float, delta_lowT),
                      SOA_SCALAR(float, zmerge),
                      SOA_SCALAR(float, dzCutOff),
                      SOA_SCALAR(float, Tpurge),
                      SOA_SCALAR(int, convergence_mode),
                      SOA_SCALAR(float, delta_highT),
                      SOA_SCALAR(float, Tstop),
                      SOA_SCALAR(float, coolingFactor),
                      SOA_SCALAR(float, vertexSize),
                      SOA_SCALAR(float, uniquetrkweight),
                      SOA_SCALAR(float, uniquetrkminp),
                      SOA_SCALAR(float, zrange))

  using ClusterParamsSoA = ClusterParams<>;

}  // namespace portablevertex

#endif  // DataFormats_PortableVertex_interface_VertexSoA_h
