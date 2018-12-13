#ifndef Fireworks_Calo_FWHGCalMultiClusterSliceSelector_h
#define Fireworks_Calo_FWHGCalMultiClusterSliceSelector_h

#include "Fireworks/Calo/interface/FWHistSliceSelector.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

class FWHGCalMultiClusterSliceSelector : public FWHistSliceSelector
{
public:
  FWHGCalMultiClusterSliceSelector(TH2F* h, const FWEventItem* i);
  ~FWHGCalMultiClusterSliceSelector() override;
 
protected:
   void getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const override; 
};

#endif
