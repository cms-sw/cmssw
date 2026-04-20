#ifndef DataFormats_VertexSoA_interface_TrackSoA_h
#define DataFormats_VertexSoA_interface_TrackSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

const int maxTotalVertex = 1024;
using TrackToVertex = Eigen::Vector<float, maxTotalVertex>; 
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

                      // scalars: one value for the whole structure
                      SOA_SCALAR(int32_t, nT),
                      SOA_SCALAR(float, totweight))

using TrackSoA = TrackSoALayout<>;

#endif  // DataFormats_VertexSoA_interface_TrackSoA_h
