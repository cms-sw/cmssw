#ifndef Fireworks_Calo_CaloUtils_h
#define Fireworks_Calo_CaloUtils_h

#include <vector>
#include "TEveVector.h"

class TEveElement;

namespace fireworks {
   void addBox( const std::vector<TEveVector> &corners, class TEveElement &oItemHolder );
   void addInvertedBox( const std::vector<TEveVector> &corners, class TEveElement &oItemHolder );
   void drawEnergyScaledBox3D( std::vector<TEveVector> &corners, float scale, class TEveElement &oItemHolder );
   void drawEnergyTower3D( std::vector<TEveVector> &corners, float scale, class TEveElement &oItemHolder );
}

#endif
