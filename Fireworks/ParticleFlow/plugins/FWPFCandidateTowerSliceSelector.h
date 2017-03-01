#ifndef Fireworks_Calo_FWPFCandTowerSliceSelector_h
#define Fireworks_Calo_FWPFCandTowerSliceSelector_h

#include "Fireworks/Calo/interface/FWHistSliceSelector.h"

class FWPFCandidateTowerSliceSelector : public FWHistSliceSelector
{
public:
   FWPFCandidateTowerSliceSelector(TH2F* h, const FWEventItem* i);
   virtual ~FWPFCandidateTowerSliceSelector();
 virtual bool aggregatePhiCells() const { return false; }

 protected:
   virtual void getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const;
};

#endif
