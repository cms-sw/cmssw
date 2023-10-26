#ifndef Geometry_HGCalCommonData_HGCalTypes_h
#define Geometry_HGCalCommonData_HGCalTypes_h

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

class HGCalTypes {
public:
  HGCalTypes() {}

  static constexpr int32_t WaferCorner0 = 0;
  static constexpr int32_t WaferCorner1 = 1;
  static constexpr int32_t WaferCorner2 = 2;
  static constexpr int32_t WaferCorner3 = 3;
  static constexpr int32_t WaferCorner4 = 4;
  static constexpr int32_t WaferCorner5 = 5;

  static constexpr int32_t UnknownPosition = -1;
  static constexpr int32_t WaferCenter = 0;
  static constexpr int32_t CornerCenterYp = 1;
  static constexpr int32_t CornerCenterYm = 2;
  static constexpr int32_t CornerCenterXp = 3;
  static constexpr int32_t CornerCenterXm = 4;
  static constexpr int32_t WaferCenterB = 5;
  static constexpr int32_t WaferCenterR = 6;

  static constexpr int32_t WaferTypeUndefined = -1;
  static constexpr int32_t WaferFineThin = 0;
  static constexpr int32_t WaferCoarseThin = 1;
  static constexpr int32_t WaferCoarseThick = 2;
  static constexpr int32_t WaferFineThick = 3;

  static constexpr int32_t WaferFull = 0;
  static constexpr int32_t WaferFive = 1;
  static constexpr int32_t WaferChopTwo = 2;
  static constexpr int32_t WaferChopTwoM = 3;
  static constexpr int32_t WaferHalf = 4;
  static constexpr int32_t WaferSemi = 5;
  static constexpr int32_t WaferSemi2 = 6;
  static constexpr int32_t WaferThree = 7;
  static constexpr int32_t WaferHalf2 = 8;
  static constexpr int32_t WaferFive2 = 9;
  static constexpr int32_t WaferLDTop = 11;
  static constexpr int32_t WaferLDBottom = 12;
  static constexpr int32_t WaferLDLeft = 13;
  static constexpr int32_t WaferLDRight = 14;
  static constexpr int32_t WaferLDFive = 15;
  static constexpr int32_t WaferLDThree = 16;
  static constexpr int32_t WaferHDTop = 21;
  static constexpr int32_t WaferHDBottom = 22;
  static constexpr int32_t WaferHDLeft = 23;
  static constexpr int32_t WaferHDRight = 24;
  static constexpr int32_t WaferHDFive = 25;
  static constexpr int32_t WaferOut = 99;
  static constexpr int32_t WaferPartLDOffset = 11;
  static constexpr int32_t WaferPartHDOffset = 21;
  static constexpr int32_t WaferTypeOffset[3] = {0, 5, 11};

  static constexpr int32_t WaferOrient0 = 0;
  static constexpr int32_t WaferOrient1 = 1;
  static constexpr int32_t WaferOrient2 = 2;
  static constexpr int32_t WaferOrient3 = 3;
  static constexpr int32_t WaferOrient4 = 4;
  static constexpr int32_t WaferOrient5 = 5;

  static constexpr int32_t WaferCenteredFront = 0;
  static constexpr int32_t WaferCenteredBack = 1;
  static constexpr int32_t CornerCenteredY = 2;
  static constexpr int32_t CornerCenteredLambda = 3;
  static constexpr int32_t WaferCenteredRotated = 4;

  static constexpr int32_t WaferCornerMin = 3;
  static constexpr int32_t WaferCornerMax = 6;
  static constexpr int32_t WaferSizeMax = 9;

  // Condition for cells in partial wafer: arr[0]*u + arr[1]*v <= arr[2]
  static constexpr std::array<int, 3> edgeWaferLDTop = {{1, 0, 7}};
  static constexpr std::array<int, 3> edgeWaferLDBottom = {{-1, 0, -8}};
  static constexpr std::array<int, 3> edgeWaferLDLeft = {{-1, 2, 8}};
  static constexpr std::array<int, 3> edgeWaferLDRight = {{1, -2, -7}};
  static constexpr std::array<int, 3> edgeWaferLDFive = {{-1, 2, 15}};
  static constexpr std::array<int, 3> edgeWaferLDThree = {{1, -2, -14}};
  static constexpr std::array<int, 3> edgeWaferHDTop = {{1, 0, 9}};
  static constexpr std::array<int, 3> edgeWaferHDBottom = {{-1, 0, -10}};
  static constexpr std::array<int, 3> edgeWaferHDLeft = {{-1, 2, 5}};
  static constexpr std::array<int, 3> edgeWaferHDRight = {{1, -2, -18}};
  static constexpr std::array<int, 3> edgeWaferHDFive = {{-1, 2, 18}};

  static constexpr int k_OffsetRotation = 10;

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

  // Packing and unpacking of type, u, v of wafers
  static int32_t packTypeUV(int type, int u, int v);
  static int32_t getUnpackedType(int id);
  static int32_t getUnpackedU(int id);
  static int32_t getUnpackedV(int id);
  // Packing and unpacking of type, u, v of wafer cells
  static int32_t packCellTypeUV(int type, int u, int v);
  static int32_t getUnpackedCellType(int id);
  static int32_t getUnpackedCellU(int id);
  static int32_t getUnpackedCellV(int id);
  // Packing and unpacking of type, cell# of wafer cells (6 inch wafers)
  static int32_t packCellType6(int type, int cell);
  static int32_t getUnpackedCellType6(int id);
  static int32_t getUnpackedCell6(int id);
  // Translate from flat file format to CMSSW format of Layer type
  static int32_t layerType(int type);
  // Get the front-back index from the layer orientation index
  static constexpr int32_t layerFrontBack(int32_t layerOrient) { return ((layerOrient == WaferCenterB) ? 1 : -1); }
  static constexpr int32_t waferFrontBack(int32_t index) { return ((index == 0) ? -1 : 1); }
  // LD vs HD and Fullvs Partial wafer
  static constexpr bool waferHD(int32_t type) { return ((type == WaferFineThin) || (type == WaferFineThick)); }
  static constexpr bool waferFull(int32_t type) { return (type == WaferFull); }

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
