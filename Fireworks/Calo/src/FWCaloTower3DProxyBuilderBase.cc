// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTower3DProxyBuilderBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 11:28:28 EST 2008
// $Id$
//

// system include files
#include <math.h>
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"

// user include files
#include "Fireworks/Calo/interface/FWCaloTower3DProxyBuilderBase.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWCaloTower3DProxyBuilderBase::FWCaloTower3DProxyBuilderBase():
m_caloData(0),
m_hist(0)
{
}

// FWCaloTower3DProxyBuilderBase::FWCaloTower3DProxyBuilderBase(const FWCaloTower3DProxyBuilderBase& rhs)
// {
//    // do actual copying here;
// }

FWCaloTower3DProxyBuilderBase::~FWCaloTower3DProxyBuilderBase()
{
}

//
// assignment operators
//
// const FWCaloTower3DProxyBuilderBase& FWCaloTower3DProxyBuilderBase::operator=(const FWCaloTower3DProxyBuilderBase& rhs)
// {
//   //An exception safe implementation is
//   FWCaloTower3DProxyBuilderBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWCaloTower3DProxyBuilderBase::addToScene(TEveElement& iContainer, TEveCaloDataHist** iCaloData)
{
   if(0==*iCaloData) {
      *iCaloData = new TEveCaloDataHist();
      
      //NOTE: must attach a histogram to TEveCaloDataHist before passing TEveCaloDataHist to TEveCalo3D
      // else we get a segmentation fault.
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      TH2F* dummy = new TH2F("background",
                        "background",
                        82, fw3dlego::xbins,
                        72, -M_PI, M_PI);
      TH1::AddDirectory(status);
      Int_t sliceIndex = (*iCaloData)->AddHistogram(dummy);
      (*iCaloData)->RefSliceInfo(sliceIndex).Setup("background", 0., 0);
      
      TEveCalo3D* calo3d = new TEveCalo3D(*iCaloData);
      calo3d->SetBarrelRadius(129);
      calo3d->SetEndCapPos(310);
      iContainer.AddElement(calo3d);
   }
   m_caloData = *iCaloData;
}

void 
FWCaloTower3DProxyBuilderBase::build(const FWEventItem* iItem,
                                     TEveElementList** product)
{
   m_towers=0;
   iItem->get(m_towers);
   if(0==m_towers) return;

   if(0==m_hist) {
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      m_hist = new TH2F(histName().c_str(),
                        (std::string("CaloTower ")+histName()+" Et distribution").c_str(),
                        82, fw3dlego::xbins,
                        72, -M_PI, M_PI);
      TH1::AddDirectory(status);
      m_sliceIndex = m_caloData->AddHistogram(m_hist);
      m_caloData->RefSliceInfo(m_sliceIndex).Setup(histName().c_str(), 0., iItem->defaultDisplayProperties().color());
   }   
   m_hist->Reset();
   
   if(0==*product) {
      //NOTE: the base class requires that something gets attached or else model changes willl not
      // be propagated to the base class
      *product = new TEveElementList();
   }
   if(iItem->defaultDisplayProperties().isVisible()) {
      m_caloData->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower) {
         (m_hist)->Fill(tower->eta(),tower->phi(), getEt(*tower));
      }
   }
   m_caloData->DataChanged();
}

void 
FWCaloTower3DProxyBuilderBase::modelChanges(const FWModelIds&, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void 
FWCaloTower3DProxyBuilderBase::applyChangesToAllModels(TEveElement* iElements)
{
   if(m_caloData && m_towers && item()) {
      m_hist->Reset();
      if(item()->defaultDisplayProperties().isVisible()) {
         
         assert(item()->size() >= m_towers->size());
         unsigned int index=0;
         for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
            if(item()->modelInfo(index).displayProperties().isVisible()) {
               (m_hist)->Fill(tower->eta(),tower->phi(), getEt(*tower));
            }
         }
      }
      m_caloData->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      m_caloData->DataChanged();
   }
}

void 
FWCaloTower3DProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
{
   FW3DDataProxyBuilder::itemBeingDestroyed(iItem);
   m_hist->Reset();
   m_caloData->DataChanged();
}

//
// const member functions
//

//
// static member functions
//
