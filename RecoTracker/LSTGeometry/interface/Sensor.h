#ifndef RecoTracker_LSTGeometry_interface_Sensor_h
#define RecoTracker_LSTGeometry_interface_Sensor_h

// Some parts of this file are guarded by the LST_STANDALONE flag to avoid depending on Eigen for standalone LST

#include <memory>
#include <unordered_map>

#ifndef LST_STANDALONE
#include <Eigen/Dense>
#endif

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

#ifndef LST_STANDALONE
  using MatrixF3x3 = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
  using MatrixF4x2 = Eigen::Matrix<float, 4, 2, Eigen::RowMajor>;
  using MatrixF4x3 = Eigen::Matrix<float, 4, 3, Eigen::RowMajor>;
  using MatrixF8x3 = Eigen::Matrix<float, 8, 3, Eigen::RowMajor>;
  using MatrixFNx2 = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;
  using MatrixFNx3 = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using RowVectorF2 = Eigen::Matrix<float, 1, 2>;
  using ColVectorF3 = Eigen::Matrix<float, 3, 1>;
  using RowVectorF3 = Eigen::Matrix<float, 1, 3>;
#endif

  using ModuleType = TrackerGeometry::ModuleType;
  using SubDetector = GeomDetEnumerators::SubDetector;
  using Location = GeomDetEnumerators::Location;

  enum SubDet { InnerPixel = 0, Barrel = 5, Endcap = 4 };
  enum Side { NegZ = 1, PosZ = 2, Center = 3 };

  struct SensorExtraData {
    // Module-level properties
    SubDetector subdet;
    Location location;
    Side side;
    unsigned int moduleId;
    unsigned int layer;
    unsigned int ring;
    bool inverted;
    // Sensor-level properties
    float centerRho;
    float centerPhi;
    bool lower;
    bool strip;
#ifndef LST_STANDALONE  // Extra data is not used in standalone, so it doesn't matter that there is a struct mismatch
    MatrixF4x3 corners;
#endif
    // Redundant, but convenient to avoid repeated computations
    float centerEta;
    float centerTheta;
    float minR;
    float maxR;
    float minZ;
    float maxZ;
    float minPhi;
    float maxPhi;
  };

  struct Sensor {
    // Info that is always needed
    ModuleType moduleType;
    float centerPhi;
    float centerX;
    float centerY;
    float centerZ;
    // Info that can be dropped after map construction
    std::unique_ptr<SensorExtraData> extra;

    Sensor() = default;
    Sensor(const Sensor&) = delete;
    Sensor& operator=(const Sensor&) = delete;
    Sensor(Sensor&&) = default;
    Sensor& operator=(Sensor&&) = default;
    Sensor(unsigned int detId,
           ModuleType moduleType,
           SubDetector subdet,
           Location location,
           Side side,
           unsigned int moduleId,
           unsigned int layer,
           unsigned int ring,
           float centerRho,
           float centerZ,
           float centerPhi,
           Surface const& surface);
  };

  using Sensors = std::unordered_map<unsigned int, Sensor>;

}  // namespace lstgeometry

#endif
