#ifndef Fireworks_Calo_CaloUtils_h
#define Fireworks_Calo_CaloUtils_h

#include <vector>

class TEveElement;
class FWProxyBuilderBase;

namespace fireworks
{
   void invertBox( std::vector<float> &corners );
   void addBox( const std::vector<float> &corners, TEveElement*,  FWProxyBuilderBase*);

   void drawEnergyScaledBox3D( const std::vector<float> &corners, float scale, TEveElement*,  FWProxyBuilderBase*, bool invert = false );
   void drawEnergyTower3D( const std::vector<float> &corners, float scale, TEveElement*, FWProxyBuilderBase*, bool reflect = false );
}

#endif
