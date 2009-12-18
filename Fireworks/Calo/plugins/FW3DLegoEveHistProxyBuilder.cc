// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoEveHistProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jul  5 11:26:11 EDT 2008
// $Id: FW3DLegoEveHistProxyBuilder.cc,v 1.1 2009/11/14 16:45:32 chrjones Exp $
//

// system include files
#include "TEveCaloData.h"
#include "TH2F.h"
#include "TEveManager.h"
#include "TEveSelection.h"

// user include files
#include "Fireworks/Calo/plugins/FW3DLegoEveHistProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DLegoEveHistProxyBuilder::FW3DLegoEveHistProxyBuilder() :
   m_hist(0), m_data(0), m_sliceIndex(-1)
{
}

// FW3DLegoEveHistProxyBuilder::FW3DLegoEveHistProxyBuilder(const FW3DLegoEveHistProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FW3DLegoEveHistProxyBuilder::~FW3DLegoEveHistProxyBuilder()
{
}

//
// assignment operators
//
// const FW3DLegoEveHistProxyBuilder& FW3DLegoEveHistProxyBuilder::operator=(const FW3DLegoEveHistProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FW3DLegoEveHistProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FW3DLegoEveHistProxyBuilder::attach(TEveElement* iElement,
                                    TEveCaloDataHist* iHist)
{
   m_data = iHist;
   if(0==m_data->GetUserData()) {
      FWFromTEveCaloDataSelector* sel = new FWFromTEveCaloDataSelector(m_data);
      //make sure it is accessible via the base class
      iHist->SetUserData(static_cast<FWFromEveSelectorBase*>(sel));
   }
}

void
FW3DLegoEveHistProxyBuilder::build()
{
   build(item(),&m_hist);
   if(0!=m_hist && -1 == m_sliceIndex) {
      m_sliceIndex = m_data->AddHistogram(m_hist);
      m_data->RefSliceInfo(m_sliceIndex).Setup(item()->name().c_str(), 0., item()->defaultDisplayProperties().color());
      
      FWFromEveSelectorBase* base = reinterpret_cast<FWFromEveSelectorBase*>(m_data->GetUserData());
      assert(0!=base);
      FWFromTEveCaloDataSelector* sel = dynamic_cast<FWFromTEveCaloDataSelector*> (base);
      assert(0!=sel);
      sel->addSliceSelector(m_sliceIndex,FWFromSliceSelector(m_hist,item()));
      
   }
   m_data->DataChanged();
}

void
FW3DLegoEveHistProxyBuilder::modelChangesImp(const FWModelIds&)
{
   //find all selected cell ids which are not from this FWEventItem and preserve only them
   // do this by moving them to the end of the list and then clearing only the end of the list
   // this avoids needing any additional memory
   TEveCaloData::vCellId_t& selected = m_data->GetCellsSelected();
   
   TEveCaloData::vCellId_t::iterator itEnd = selected.end();
   for(TEveCaloData::vCellId_t::iterator it = selected.begin();
       it != itEnd;
       ++it) {
      if(it->fSlice ==m_sliceIndex) {
         //we have found one we want to get rid of, so we swap it with the
         // one closest to the end which is not of this slice
         do {
            TEveCaloData::vCellId_t::iterator itLast = itEnd-1;
            itEnd = itLast;
         } while (itEnd != it && itEnd->fSlice==m_sliceIndex);
         
         if(itEnd != it) {
            std::swap(*it,*itEnd);
         } else {
            //shouldn't go on since advancing 'it' will put us past itEnd
            break;
         }
         //std::cout <<"keeping "<<it->fTower<<" "<<it->fSlice<<std::endl;
      }
   }
   selected.erase(itEnd,selected.end());
      
   applyChangesToAllModels();
   
   if (!selected.empty()) {
      if(0==m_data->GetSelectedLevel()) {
         gEve->GetSelection()->AddElement(m_data);
      }         
   } else {
      if(1==m_data->GetSelectedLevel()||2==m_data->GetSelectedLevel()) {
         gEve->GetSelection()->RemoveElement(m_data);
      }
   }
   
   m_data->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
   m_data->CellSelectionChanged();
   m_data->DataChanged();
}

void
FW3DLegoEveHistProxyBuilder::addToSelect(double iEta, double iPhi)
{
   TEveCaloData::vCellId_t& selected = m_data->GetCellsSelected();
   //NOTE: I tried calling TEveCalo::GetCellList but it always returned 0, probably because of threshold issues
   // but looking at the TEveCaloHist::GetCellList code the CellId_t is just the histograms bin # and the slice
   
   selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(iEta,iPhi),m_sliceIndex));   
}

void
FW3DLegoEveHistProxyBuilder::itemChangedImp(const FWEventItem*)
{

}

void
FW3DLegoEveHistProxyBuilder::itemBeingDestroyedImp(const FWEventItem* iItem)
{
   m_hist->Reset();
   m_data->DataChanged();
}


//
// const member functions
//

//
// static member functions
//
