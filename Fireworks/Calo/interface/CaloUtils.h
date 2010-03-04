#ifndef Fireworks_Calo_CaloUtils_h
#define Fireworks_Calo_CaloUtils_h

namespace fireworks {
   void drawCaloHit3D(std::vector<TEveVector> &corners, const FWEventItem* iItem, class TEveElement &oItemHolder, Float_t scaleFraction);
   void drawEcalHit3D(std::vector<TEveVector> &corners, const FWEventItem* iItem, class TEveElement &oItemHolder, Float_t scale);
}

#endif
