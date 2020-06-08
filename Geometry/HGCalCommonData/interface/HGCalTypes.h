#ifndef Geometry_HGCalCommonData_HGCalTypes_h
#define Geometry_HGCalCommonData_HGCalTypes_h

#include <cmath>
#include <cstdint>
#include <vector>

class HGCalTypes {
public:
  HGCalTypes() {}

  enum WaferCorner {
    WaferCorner0 = 0,
    WaferCorner1 = 1,
    WaferCorner2 = 2,
    WaferCorner3 = 3,
    WaferCorner4 = 4,
    WaferCorner5 = 5
  };

  enum WaferPosition {
    UnknownPosition = -1,
    WaferCenter = 0,
    CornerCenterYp = 1,
    CornerCenterYm = 2,
    CornerCenterXp = 3,
    CornerCenterXm = 4
  };

  enum WaferType { WaferFineThin = 0, WaferCoarseThin = 1, WaferCoarseThick = 2, WaferFineThick = 3 };

  enum WaferSizeType {
    WaferFull = 0,
    WaferFive = 1,
    WaferChopTwo = 2,
    WaferChopTwoM = 3,
    WaferHalf = 4,
    WaferSemi = 5,
    WaferSemi2 = 6,
    WaferThree = 7,
    WaferOut = 99
  };

  enum TileType { TileFine = 0, TileCoarseCast = 1, TileCoarseMould = 2 };

  enum TileSiPMType { SiPMUnknown = 0, SiPMSmall = 2, SiPMLarge = 4 };
};

#endif
