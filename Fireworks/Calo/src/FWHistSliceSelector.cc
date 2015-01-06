#include "Fireworks/Calo/interface/FWHistSliceSelector.h"

#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "TEveCaloData.h"
#include "TH2F.h"
#include "Rtypes.h"

FWHistSliceSelector::FWHistSliceSelector(TH2F* h, const FWEventItem* item):
   FWFromSliceSelector(item)
{
   m_hist = h;
}


FWHistSliceSelector::~FWHistSliceSelector()
{}

bool
FWHistSliceSelector::matchCell(const TEveCaloData::CellId_t& iCell, int itemIdx) const
{
   float eta, phi;
   getItemEntryEtaPhi(itemIdx, eta, phi);

   int idx = m_hist->FindBin(eta, phi);
   int nBinsX = m_hist->GetXaxis()->GetNbins() + 2;

   int etaBin, phiBin, w, newPhiBin;
   m_hist->GetBinXYZ(idx, etaBin, phiBin, w);

   if (aggregatePhiCells()) {
      bool match= false;
      if (TMath::Abs(eta) > 4.716)
      {
         newPhiBin =  ((phiBin + 1) / 4) * 4 - 1;
         if (newPhiBin <= 0) newPhiBin = 71;

         idx = etaBin + newPhiBin*nBinsX;
         match |= (idx == iCell.fTower);

         idx += nBinsX;
         match |= (idx == iCell.fTower);

         idx += nBinsX;
         if (newPhiBin == 71)
            idx = etaBin + 1*nBinsX;
         match |= (idx == iCell.fTower);

         idx += nBinsX;
         match |= (idx == iCell.fTower);
      } 
      else if (TMath::Abs(eta) > 1.873)
      {
         newPhiBin =  ((phiBin  + 1) / 2) * 2 -1;
         idx = etaBin + newPhiBin*nBinsX;
         match = ( idx == iCell.fTower ||  idx + nBinsX == iCell.fTower);
      }
      else
      {
         match = ( idx == iCell.fTower);
      }
   
      return match;
   }
   else 
   {
      return idx == iCell.fTower;
   }
}

void
FWHistSliceSelector::doSelect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;

   FWChangeSentry sentry(*(m_item->changeManager()));
   size_t size = m_item->size();
   for (size_t index =0; index < size; ++index)
   {
      if (m_item->modelInfo(index).m_displayProperties.isVisible() && !m_item->modelInfo(index).isSelected())
      {
         if (matchCell(iCell, index))
         {
            m_item->select(index);
            break;
         }
      }
   }
}

void
FWHistSliceSelector::doUnselect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;
  
   //  std::cout <<"  doUnselect "<<std::endl;

   FWChangeSentry sentry(*(m_item->changeManager()));

   size_t size = m_item->size();
   for (size_t index =0; index < size; ++index)
   {
      if ( m_item->modelInfo(index).m_displayProperties.isVisible() &&
           m_item->modelInfo(index).isSelected()) {
         if (matchCell(iCell, index))
         {
            //  std::cout <<"  doUnselect "<<index<<std::endl;
            m_item->unselect(index);
            break;
         }
      }
   }
}
