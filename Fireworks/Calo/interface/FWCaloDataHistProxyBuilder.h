#ifndef Fireworks_Calo_FWCaloHistDataProxyBuilder_h
#define Fireworks_Calo_FWCaloHistDataProxyBuilder_h


#include "Fireworks/Calo/interface/FWCaloDataProxyBuilderBase.h"
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"

class TH2F;
class FWHistSliceSelector;

class FWCaloDataHistProxyBuilder : public FWCaloDataProxyBuilderBase
{
public:
   FWCaloDataHistProxyBuilder();
   virtual ~FWCaloDataHistProxyBuilder();

protected:
   virtual bool assertCaloDataSlice();
   virtual FWHistSliceSelector*  instantiateSliceSelector() = 0;
   virtual void itemBeingDestroyed(const FWEventItem*);
   virtual void setCaloData(const fireworks::Context&);
   void addEntryToTEveCaloData(float eta, float phi, float Et, bool isSelected);

   TH2F* m_hist;
   FWHistSliceSelector* m_sliceSelector;
};



#endif

