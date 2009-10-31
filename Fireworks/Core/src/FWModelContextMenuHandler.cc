// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelContextMenuHandler
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Sep 22 13:26:04 CDT 2009
// $Id: FWModelContextMenuHandler.cc,v 1.2 2009/09/24 00:43:07 chrjones Exp $
//

// system include files
#include <assert.h>
#include "TGMenu.h"

// user include files
#include "Fireworks/Core/src/FWModelContextMenuHandler.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"

//
// constants, enums and typedefs
//
enum MenuOptions {
   kSetVisibleMO,
   kSetColorMO,
   kOpenDetailViewMO,
   kAfterOpenDetailViewMO,
   kOpenObjectControllerMO=100,
   kOpenCollectionControllerMO,
   kLastOfMO
};

//
// static data member definitions
//
static const char* const kOpenDetailView = "Open Detailed View ...";

//
// constructors and destructor
//
FWModelContextMenuHandler::FWModelContextMenuHandler(FWSelectionManager* iSM,
                                                     FWDetailViewManager* iDVM,
                                                     FWColorManager* iCM,
                                                     FWGUIManager* iGM):
m_modelPopup(0),
m_colorPopup(0),
m_selectionManager(iSM),
m_detailViewManager(iDVM),
m_colorManager(iCM),
m_guiManager(iGM),
m_nDetailViewChoices(0)
{
}

// FWModelContextMenuHandler::FWModelContextMenuHandler(const FWModelContextMenuHandler& rhs)
// {
//    // do actual copying here;
// }

FWModelContextMenuHandler::~FWModelContextMenuHandler()
{
   delete m_modelPopup;
}

//
// assignment operators
//
// const FWModelContextMenuHandler& FWModelContextMenuHandler::operator=(const FWModelContextMenuHandler& rhs)
// {
//   //An exception safe implementation is
//   FWModelContextMenuHandler temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
namespace  {
   class change_visibility {
   public:
      change_visibility(bool iIsVisible): m_isVisible(iIsVisible) {}
      void operator()(const FWModelId& iID) const {
         FWDisplayProperties p = iID.item()->modelInfo(iID.index()).displayProperties();
         p.setIsVisible(m_isVisible);
         iID.item()->setDisplayProperties(iID.index(),p);
      }
      bool m_isVisible; 
   };
}
void 
FWModelContextMenuHandler::chosenItem(Int_t iChoice)
{
   assert(!m_selectionManager->selected().empty());
   switch (iChoice) {
      case kSetVisibleMO:
      {
         FWModelId id = *(m_selectionManager->selected().begin());
         const FWDisplayProperties& props = id.item()->modelInfo(id.index()).displayProperties();
         for_each(m_selectionManager->selected().begin(),
                  m_selectionManager->selected().end(), 
                  change_visibility(!props.isVisible())
                  );
         break;
      }
      case kSetColorMO:
      {
         createColorPopup();
         m_colorPopup->SetName("Selected");
         std::vector<Pixel_t> colors;
         for(unsigned int index=0; index <m_colorManager->numberOfIndicies(); ++index) {
            colors.push_back((Pixel_t)gVirtualX->GetPixel(m_colorManager->indexToColor(index)));
         }
         m_colorPopup->ResetColors(colors, m_colorManager->backgroundColorIndex()==FWColorManager::kBlackIndex);
         FWModelId id = *(m_selectionManager->selected().begin());
         m_colorPopup->SetSelection(gVirtualX->GetPixel(id.item()->modelInfo(id.index()).displayProperties().color()));
         m_colorPopup->PlacePopup(m_x, m_y, m_colorPopup->GetDefaultWidth(), m_colorPopup->GetDefaultHeight());
         
         break;
      }
      case kOpenObjectControllerMO:
      {
         m_guiManager->showModelPopup();
         break;
      }
      case kOpenCollectionControllerMO:
      {
         m_guiManager->showEDIFrame();
         break;
      }
      case kOpenDetailViewMO:
      default:
      {
         assert(iChoice<kOpenObjectControllerMO);
         assert(m_selectionManager->selected().size()==1);
         std::vector<std::string> viewChoices = m_detailViewManager->detailViewsFor(*(m_selectionManager->selected().begin()));
         assert(0!=viewChoices.size());
         m_detailViewManager->openDetailViewFor(*(m_selectionManager->selected().begin()),viewChoices[iChoice-kOpenDetailViewMO]) ;
         break;
      }
         break;
   }
}
void 
FWModelContextMenuHandler::colorChangeRequested(Int_t iIndex)
{
   Color_t color =m_colorManager->indexToColor(iIndex);

   for(std::set<FWModelId>::const_iterator it =m_selectionManager->selected().begin(),
       itEnd = m_selectionManager->selected().end();
       it != itEnd;
       ++it) {
      const FWDisplayProperties changeProperties(color, it->item()->modelInfo(it->index()).displayProperties().isVisible());
      it->item()->setDisplayProperties(it->index(),changeProperties);
   }
}


//
// const member functions
//
void 
FWModelContextMenuHandler::showSelectedModelContext(Int_t iX, Int_t iY) const
{
   assert(!m_selectionManager->selected().empty());
   createModelContext();

   //setup the menu based on this object
   FWModelId id = *(m_selectionManager->selected().begin());
   const FWDisplayProperties& props = id.item()->modelInfo(id.index()).displayProperties();
   if(props.isVisible()) {
      m_modelPopup->CheckEntry(kSetVisibleMO);
   }else {
      m_modelPopup->UnCheckEntry(kSetVisibleMO);
   }

   if(m_selectionManager->selected().size()==1) {
      std::vector<std::string> viewChoices = m_detailViewManager->detailViewsFor(*(m_selectionManager->selected().begin()));
      //CDJ NEED TO ADD ALL CHOICES HERE
      if(viewChoices.size()>0) {
         if(m_nDetailViewChoices != viewChoices.size()) {
            if(m_nDetailViewChoices>viewChoices.size()) {
               for(unsigned int index = m_nDetailViewChoices; index != viewChoices.size(); --index) {
                  m_modelPopup->DeleteEntry(kOpenDetailViewMO+index);
               }
            } else {
               if(not (m_nDetailViewChoices==0 && viewChoices.size()==1)) {
                  for(unsigned int index = m_nDetailViewChoices; index != viewChoices.size(); ++index) {
                     m_modelPopup->AddEntry(kOpenDetailView,kOpenDetailViewMO+index);
                  }
               }
            }
         }
         m_nDetailViewChoices = viewChoices.size();
         const std::string kStart("Open ");
         const std::string kEnd(" Detail View ...");
         for(unsigned int index=0; index != m_nDetailViewChoices; ++index) {
            m_modelPopup->GetEntry(index+kOpenDetailViewMO)->GetLabel()->SetString((kStart+viewChoices[index]+kEnd).c_str());
         }
         m_modelPopup->EnableEntry(kOpenDetailViewMO);
      } else {
         for(int i =m_nDetailViewChoices-1; i > 0; --i) {
            m_modelPopup->DeleteEntry(kOpenDetailViewMO+i);
         }
         m_nDetailViewChoices=0;
         m_modelPopup->GetEntry(kOpenDetailViewMO)->GetLabel()->SetString(kOpenDetailView);
         m_modelPopup->HideEntry(kOpenDetailViewMO);
      }
   } else {
      m_nDetailViewChoices=0;
      for(int i =m_nDetailViewChoices-1; i > 0; --i) {
         m_modelPopup->DeleteEntry(kOpenDetailViewMO+i);
      }
      m_nDetailViewChoices=0;
      m_modelPopup->GetEntry(kOpenDetailViewMO)->GetLabel()->SetString(kOpenDetailView);
      m_modelPopup->HideEntry(kOpenDetailViewMO);
   }
   m_x=iX;
   m_y=iY;
   m_modelPopup->PlaceMenu(iX,iY,false,true);
}

void 
FWModelContextMenuHandler::createModelContext() const
{
   if(0==m_modelPopup) {
      m_modelPopup = new TGPopupMenu();
      
      m_modelPopup->AddEntry("Set Visible",kSetVisibleMO);
      m_modelPopup->AddEntry("Set Color ...",kSetColorMO);
      m_modelPopup->AddEntry(kOpenDetailView,kOpenDetailViewMO);
      m_modelPopup->AddSeparator();
      m_modelPopup->AddEntry("Open Object Controller ...",kOpenObjectControllerMO);
      m_modelPopup->AddEntry("Open Collection Controller ...",kOpenCollectionControllerMO);

      m_modelPopup->Connect("Activated(Int_t)",
                            "FWModelContextMenuHandler",
                            const_cast<FWModelContextMenuHandler*>(this),
                            "chosenItem(Int_t)");
      
   }
}

void 
FWModelContextMenuHandler::createColorPopup() const
{
   if(0==m_colorPopup) {
      std::vector<Pixel_t> colors;
      for(unsigned int index=0; index <m_colorManager->numberOfIndicies(); ++index) {
         colors.push_back((Pixel_t)gVirtualX->GetPixel(m_colorManager->indexToColor(index)));
      }
      
      //Pixel_t selection = gVirtualX->GetPixel(m_collection->defaultDisplayProperties().color());
      
      m_colorPopup = new FWColorPopup(gClient->GetDefaultRoot(), colors.front());
      m_colorPopup->InitContent("", colors);
      m_colorPopup->Connect("ColorBookkeeping(Int_t)","FWModelContextMenuHandler", const_cast<FWModelContextMenuHandler*>(this), "colorChangeRequested(Int_t)");      
   }
}

//
// static member functions
//

ClassImp(FWModelContextMenuHandler)
