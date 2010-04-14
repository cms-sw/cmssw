#ifndef Fireworks_Calo_CaloUtils_h
#define Fireworks_Calo_CaloUtils_h

namespace fireworks {
   void addBox(const std::vector<TEveVector> &corners, class TEveElement &oItemHolder);
   void addInvertedBox(const std::vector<TEveVector> &corners, class TEveElement &oItemHolder);
   void drawEnergyScaledBox3D(std::vector<TEveVector> &corners, Float_t scale, class TEveElement &oItemHolder);
   void drawTransverseEnergyScaledBox3D(std::vector<TEveVector> &corners, Float_t scale, class TEveElement &oItemHolder);
   void drawEnergyTower3D(std::vector<TEveVector> &corners, Float_t scale, class TEveElement &oItemHolder);
   void drawTransverseEnergyTower3D(std::vector<TEveVector> &corners, Float_t scale, class TEveElement &oItemHolder);
}

#endif
