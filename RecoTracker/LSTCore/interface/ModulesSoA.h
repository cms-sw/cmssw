#ifndef RecoTracker_LSTCore_interface_ModulesSoA_h
#define RecoTracker_LSTCore_interface_ModulesSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  enum SubDet { InnerPixel = 0, Barrel = 5, Endcap = 4 };

  enum Side { NegZ = 1, PosZ = 2, Center = 3 };

  enum ModuleType { PS, TwoS, PixelModule };

  enum ModuleLayerType { Pixel, Strip, InnerPixelLayer };

  GENERATE_SOA_LAYOUT(ModulesSoALayout,
                      SOA_COLUMN(unsigned int, detIds),
                      SOA_COLUMN(Params_Modules::ArrayU16xMaxConnected, moduleMap),
                      SOA_COLUMN(unsigned int, mapdetId),
                      SOA_COLUMN(uint16_t, mapIdx),
                      SOA_COLUMN(uint16_t, nConnectedModules),
                      SOA_COLUMN(float, drdzs),
                      SOA_COLUMN(float, dxdys),
                      SOA_COLUMN(uint16_t, partnerModuleIndices),
                      SOA_COLUMN(short, layers),
                      SOA_COLUMN(short, rings),
                      SOA_COLUMN(short, modules),
                      SOA_COLUMN(short, rods),
                      SOA_COLUMN(short, subdets),
                      SOA_COLUMN(short, sides),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, r),
                      SOA_COLUMN(bool, isInverted),
                      SOA_COLUMN(bool, isLower),
                      SOA_COLUMN(bool, isAnchor),
                      SOA_COLUMN(ModuleType, moduleType),
                      SOA_COLUMN(ModuleLayerType, moduleLayerType),
                      SOA_COLUMN(int, lstLayers),
                      SOA_SCALAR(uint16_t, nModules),
                      SOA_SCALAR(uint16_t, nLowerModules))

  GENERATE_SOA_LAYOUT(ModulesPixelSoALayout, SOA_COLUMN(unsigned int, connectedPixels))

  using ModulesSoA = ModulesSoALayout<>;
  using ModulesPixelSoA = ModulesPixelSoALayout<>;

  using Modules = ModulesSoA::View;
  using ModulesConst = ModulesSoA::ConstView;
  using ModulesPixel = ModulesPixelSoA::View;
  using ModulesPixelConst = ModulesPixelSoA::ConstView;

}  // namespace lst

#endif
