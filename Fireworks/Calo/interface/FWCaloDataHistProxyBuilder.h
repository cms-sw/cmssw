#ifndef Fireworks_Calo_FWCaloHistDataProxyBuilder_h
#define Fireworks_Calo_FWCaloHistDataProxyBuilder_h

#include "Fireworks/Calo/interface/FWCaloDataProxyBuilderBase.h"
#include "Fireworks/Calo/interface/FWFromTEveCaloDataSelector.h"

class TH2F;
class FWHistSliceSelector;

class FWCaloDataHistProxyBuilder : public FWCaloDataProxyBuilderBase {
public:
  FWCaloDataHistProxyBuilder();
  ~FWCaloDataHistProxyBuilder() override;

protected:
  bool assertCaloDataSlice() override;
  virtual FWHistSliceSelector* instantiateSliceSelector() = 0;
  void itemBeingDestroyed(const FWEventItem*) override;
  void setCaloData(const fireworks::Context&) override;
  void addEntryToTEveCaloData(float eta, float phi, float Et, bool isSelected);

  TH2F* m_hist;
  FWHistSliceSelector* m_sliceSelector;
};

#endif
