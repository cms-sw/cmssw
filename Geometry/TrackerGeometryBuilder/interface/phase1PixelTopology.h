#ifndef Geometry_TrackerGeometryBuilder_phase1PixelTopology_h
#define Geometry_TrackerGeometryBuilder_phase1PixelTopology_h

#include <cstdint>

namespace phase1PixelTopology {

  constexpr uint16_t numRowsInRoc     = 80;
  constexpr uint16_t numColsInRoc     = 52;
  constexpr uint16_t lastRowInRoc     = numRowsInRoc - 1;
  constexpr uint16_t lastColInRoc     = numColsInRoc - 1;

  constexpr uint16_t numRowsInModule  = 2 * numRowsInRoc;
  constexpr uint16_t numColsInModule  = 8 * numColsInRoc;
  constexpr uint16_t lastRowInModule  = numRowsInModule - 1;
  constexpr uint16_t lastColInModule  = numColsInModule - 1;

  constexpr int16_t xOffset = -81;
  constexpr int16_t yOffset = -54*4;

  constexpr uint32_t numPixsInModule = uint32_t(numRowsInModule)* uint32_t(numColsInModule);

  constexpr uint32_t numberOfModules = 1856;
  
  constexpr uint32_t layerStart[11] = {0,96,320,672,1184,1296,1408,1520,1632,1744,1856};
  constexpr char const * layerName[10] = {"BL1","BL2","BL3","BL4",
   		  	                  "E+1", "E+2", "E+3",
			                  "E-1", "E-2", "E-3"
                                          };


  // this is for the ROC n<512 (upgrade 1024)
  constexpr inline
  uint16_t  divu52(uint16_t n) {
    n = n>>2;
    uint16_t q = (n>>1) + (n>>4);
    q = q + (q>>4) + (q>>5); q = q >> 3;
    uint16_t r = n - q*13;
    return q + ((r + 3) >> 4);
  }

  constexpr inline
  bool isEdgeX(uint16_t px) { return (px==0) | (px==lastRowInModule);}
  constexpr inline
  bool isEdgeY(uint16_t py) { return (py==0) | (py==lastColInModule);}


  constexpr inline
  uint16_t toRocX(uint16_t px) { return (px<numRowsInRoc) ? px : px-numRowsInRoc; }
  constexpr inline
  uint16_t toRocY(uint16_t py) {
    auto roc = divu52(py);
    return py - 52*roc;
  }

  constexpr inline
  bool isBigPixX(uint16_t px) {
    return (px==79) | (px==80);
  }

  constexpr inline
  bool isBigPixY(uint16_t py) {
    auto ly=toRocY(py);
    return (ly==0) | (ly==lastColInRoc);
  }

  constexpr inline
  uint16_t localX(uint16_t px) {
    auto shift = 0;
    if (px>lastRowInRoc) shift+=1;
    if (px>numRowsInRoc) shift+=1;
    return px+shift;
  }

  constexpr inline
  uint16_t localY(uint16_t py) {
    auto roc = divu52(py);
    auto shift = 2*roc;
    auto yInRoc = py - 52*roc;
    if (yInRoc>0) shift+=1;
    return py+shift;
  }

}

#endif // Geometry_TrackerGeometryBuilder_phase1PixelTopology_h
