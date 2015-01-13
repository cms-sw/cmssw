#ifndef Fireworks_Calo_FWHistSliceSelector_h
#define Fireworks_Calo_FWHistSliceSelector_h

#include "Fireworks/Calo/src/FWFromSliceSelector.h"
class TH2F;

class FWHistSliceSelector : public FWFromSliceSelector
{
public:
   FWHistSliceSelector(TH2F* h, const FWEventItem* item);
   virtual  ~FWHistSliceSelector();

   virtual void doSelect(const TEveCaloData::CellId_t&);
   virtual void doUnselect(const TEveCaloData::CellId_t&);

   virtual bool aggregatePhiCells() const { return true; }

protected: 
   virtual void getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const = 0;
   bool matchCell(const TEveCaloData::CellId_t& iCell, int idx) const;

   TH2F* m_hist;
};


#endif
