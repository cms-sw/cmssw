#ifndef Fireworks_Calo_CaloUtils_h
#define Fireworks_Calo_CaloUtils_h

#include <vector>
#include "TEveVector.h"

class TEveElement;
class FWProxyBuilderBase;

namespace fireworks {
void addBox( const std::vector<TEveVector> &corners, TEveElement*,  FWProxyBuilderBase*);
   void addInvertedBox( const std::vector<TEveVector> &corners, TEveElement*,  FWProxyBuilderBase*);
   void drawEnergyScaledBox3D( std::vector<TEveVector> &corners, float scale, TEveElement*,  FWProxyBuilderBase*);
   void drawEnergyTower3D( std::vector<TEveVector> &corners, float scale, TEveElement*, FWProxyBuilderBase*);
}

#endif
