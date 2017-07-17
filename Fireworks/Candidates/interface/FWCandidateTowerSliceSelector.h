#ifndef Fireworks_Calo_FWCandTowerSliceSelector_h
#define Fireworks_Calo_FWCandTowerSliceSelector_h

#include "Fireworks/Calo/interface/FWHistSliceSelector.h"
class FWSimpleProxyHelper;

class FWCandidateTowerSliceSelector : public FWHistSliceSelector
{
public:
   FWCandidateTowerSliceSelector(TH2F* h, const FWEventItem* i,  FWSimpleProxyHelper* m_helper);
   virtual ~FWCandidateTowerSliceSelector();

   virtual bool aggregatePhiCells() const { return false; }
 protected:
   virtual void getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const;
   FWSimpleProxyHelper* m_helper;
};

#endif
