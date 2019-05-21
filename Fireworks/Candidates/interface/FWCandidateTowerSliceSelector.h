#ifndef Fireworks_Calo_FWCandTowerSliceSelector_h
#define Fireworks_Calo_FWCandTowerSliceSelector_h

#include "Fireworks/Calo/interface/FWHistSliceSelector.h"
class FWSimpleProxyHelper;

class FWCandidateTowerSliceSelector : public FWHistSliceSelector {
public:
  FWCandidateTowerSliceSelector(TH2F* h, const FWEventItem* i, FWSimpleProxyHelper* m_helper);
  ~FWCandidateTowerSliceSelector() override;

  bool aggregatePhiCells() const override { return false; }

protected:
  void getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const override;
  FWSimpleProxyHelper* m_helper;
};

#endif
