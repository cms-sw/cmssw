#ifndef Fireworks_Calo_FWPFCandTowerSliceSelector_h
#define Fireworks_Calo_FWPFCandTowerSliceSelector_h

#include "Fireworks/Calo/interface/FWHistSliceSelector.h"

class FWPFCandidateTowerSliceSelector : public FWHistSliceSelector {
public:
  FWPFCandidateTowerSliceSelector(TH2F* h, const FWEventItem* i);
  ~FWPFCandidateTowerSliceSelector() override;
  bool aggregatePhiCells() const override { return false; }

protected:
  void getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const override;
};

#endif
