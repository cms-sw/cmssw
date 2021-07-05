#ifndef Geometry_HGCalCommonData_HGCalTypes_h
#define Geometry_HGCalCommonData_HGCalTypes_h

#include <cmath>
#include <cstdint>
#include <vector>

class HGCalTypes {
public:
  HGCalTypes() {}

  enum class CellType {
    UndefinedType = -1,
    CentralType = 0,
    BottomLeftEdge = 1,
    LeftEdge = 2,
    TopLeftEdge = 3,
    TopRightEdge = 4,
    RightEdge = 5,
    BottomRightEdge = 6,
    BottomCorner = 11,
    BottomLeftCorner = 12,
    TopLeftCorner = 13,
    TopCorner = 14,
    TopRightCorner = 15,
    BottomRightCorner = 16
  };

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
    CornerCenterXm = 4,
    WaferCenterB = 5,
    WaferCenterR = 6
  };

  enum WaferType {
    WaferTypeUndefined = -1,
    WaferFineThin = 0,
    WaferCoarseThin = 1,
    WaferCoarseThick = 2,
    WaferFineThick = 3
  };

  enum WaferPartialType {
    WaferFull = 0,
    WaferFive = 1,
    WaferChopTwo = 2,
    WaferChopTwoM = 3,
    WaferHalf = 4,
    WaferSemi = 5,
    WaferSemi2 = 6,
    WaferThree = 7,
    WaferHalf2 = 8,
    WaferFive2 = 9,
    WaferLDTop = 11,
    WaferLDBottom = 12,
    WaferLDLeft = 13,
    WaferLDRight = 14,
    WaferLDFive = 15,
    WaferLDTree = 16,
    WaferHDTop = 21,
    WaferHDBottom = 22,
    WaferHDLeft = 23,
    WaferHDRight = 24,
    WaferHDFive = 25,
    WaferOut = 99
  };

  enum LayerType {
    WaferCenteredFront = 0,
    WaferCenteredBack = 1,
    CornerCenteredY = 2,
    CornerCenteredLambda = 3,
    WaferCenteredRotated = 4
  };

  static constexpr int32_t WaferCornerMin = 3;
  static constexpr int32_t WaferCornerMax = 6;
  static constexpr int32_t WaferSizeMax = 7;

  static constexpr double c00 = 0.0;
  static constexpr double c22 = 0.225;
  static constexpr double c25 = 0.25;
  static constexpr double c27 = 0.275;
  static constexpr double c50 = 0.5;
  static constexpr double c61 = 0.6125;
  static constexpr double c75 = 0.75;
  static constexpr double c77 = 0.775;
  static constexpr double c88 = 0.8875;
  static constexpr double c10 = 1.0;

  enum TileType { TileFine = 0, TileCoarseCast = 1, TileCoarseMould = 2 };

  enum TileSiPMType { SiPMUnknown = 0, SiPMSmall = 2, SiPMLarge = 4 };

  static int32_t packTypeUV(int type, int u, int v);
  static int32_t getUnpackedType(int id);
  static int32_t getUnpackedU(int id);
  static int32_t getUnpackedV(int id);
  static int32_t packCellTypeUV(int type, int u, int v);
  static int32_t getUnpackedCellType(int id);
  static int32_t getUnpackedCellU(int id);
  static int32_t getUnpackedCellV(int id);
  static int32_t packCellType6(int type, int cell);
  static int32_t getUnpackedCellType6(int id);
  static int32_t getUnpackedCell6(int id);

private:
  static constexpr int32_t facu_ = 1;
  static constexpr int32_t facv_ = 100;
  static constexpr int32_t factype_ = 1000000;
  static constexpr int32_t signu_ = 10000;
  static constexpr int32_t signv_ = 100000;
  static constexpr int32_t maxuv_ = 100;
  static constexpr int32_t maxsign_ = 10;
  static constexpr int32_t maxtype_ = 10;
  static constexpr int32_t faccell_ = 100;
  static constexpr int32_t faccelltype_ = 10000;
  static constexpr int32_t faccell6_ = 1000;
};

#endif
