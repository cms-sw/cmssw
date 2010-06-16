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
// $Id: FWModelContextMenuHandler.cc,v 1.14 2010/06/15 17:28:29 matevz Exp $
//

// system include files
#include <cassert>
#include "TColor.h"
#include "TGColorDialog.h"
#include "TGMenu.h"
#include "KeySymbols.h"

// user include files
#include "Fireworks/Core/src/FWModelContextMenuHandler.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWViewContextMenuHandlerBase.h"

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
   kViewOptionsMO=1000,
   kLastOfMO
};


class FWPopupMenu : public TGPopupMenu
{
public:
   FWPopupMenu(const TGWindow* p=0, UInt_t w=10, UInt_t h=10, UInt_t options=0) :
      TGPopupMenu(p, w, h, options)
   {
      AddInput(kKeyPressMask);
   }

   // virtual void	PlaceMenu(Int_t x, Int_t y, Bool_t stick_mode, Bool_t grab_pointer)
   // {
   //    TGPopupMenu::PlaceMenu(x, y, stick_mode, grab_pointer);
   //    gVirtualX->GrabKey(fId, 0l, kAnyModifier, kTRUE);
   // }

   virtual void PoppedUp()
   {
      TGPopupMenu::PoppedUp();
      gVirtualX->SetInputFocus(fId);
      gVirtualX->GrabKey(fId, 0l, kAnyModifier, kTRUE);
      
   }

   virtual void PoppedDown()
   {
      gVirtualX->GrabKey(fId, 0l, kAnyModifier, kFALSE);
      TGPopupMenu::PoppedDown();
   }

   virtual Bool_t HandleKey(Event_t* event)
   {
      if (event->fType != kGKeyPress) return kTRUE;

      UInt_t keysym;
      char tmp[2];
      gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);

      TGMenuEntry *ce = fCurrent;

      switch (keysym)
      {
         case kKey_Up:
         {
            if (ce) ce = (TGMenuEntry*)GetListOfEntries()->Before(ce);
            while (ce && ((ce->GetType() == kMenuSeparator) ||
                          (ce->GetType() == kMenuLabel) ||
                          !(ce->GetStatus() & kMenuEnableMask)))
            {
               ce = (TGMenuEntry*)GetListOfEntries()->Before(ce);
            }
            if (!ce) ce = (TGMenuEntry*)GetListOfEntries()->Last();
            Activate(ce);
            break;
         }
         case kKey_Down:
         {
            if (ce) ce = (TGMenuEntry*)GetListOfEntries()->After(ce);
            while (ce && ((ce->GetType() == kMenuSeparator) ||
                          (ce->GetType() == kMenuLabel) ||
                          !(ce->GetStatus() & kMenuEnableMask)))
            {
               ce = (TGMenuEntry*)GetListOfEntries()->After(ce);
            }
            if (!ce) ce = (TGMenuEntry*)GetListOfEntries()->First();
            Activate(ce);
            break;
         }
         case kKey_Enter:
         case kKey_Return:
         {
            Event_t ev;
            ev.fType = kButtonRelease;
            ev.fWindow = fId;
            return HandleButton(&ev);
         }
         case kKey_Escape:
         {
            fCurrent = 0;
            void *dummy = 0;
            return EndMenu(dummy);
         }
         default:
         {
            break;
         }
      }

      return kTRUE;
   }
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
m_seperator(0),
m_viewSeperator(0),
m_nDetailViewEntries(0),
m_nViewEntries(0),
m_viewHander(0)
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
         iID.item()->setDisplayProperties(iID.index(), p);
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
         FWModelId id = *(m_selectionManager->selected().begin());
         createColorPopup();
         m_colorPopup->SetName("Selected");
         std::vector<Color_t> colors;
         m_colorManager->fillLimitedColors(colors);
         m_colorPopup->ResetColors(colors, m_colorManager->backgroundColorIndex()==FWColorManager::kBlackIndex);
         m_colorPopup->SetSelection(id.item()->modelInfo(id.index()).displayProperties().color());
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
      case kViewOptionsMO:
      default:
      {
         if(iChoice>=kViewOptionsMO) {
            assert(0!=m_viewHander);
            m_viewHander->select(iChoice-kViewOptionsMO, *(m_selectionManager->selected().begin()), m_x, m_y);
         }else {
            assert(iChoice<kOpenObjectControllerMO);
            assert(m_selectionManager->selected().size()==1);
            std::vector<std::string> viewChoices = m_detailViewManager->detailViewsFor(*(m_selectionManager->selected().begin()));
            assert(0!=viewChoices.size());
            m_detailViewManager->openDetailViewFor(*(m_selectionManager->selected().begin()),viewChoices[iChoice-kOpenDetailViewMO]) ;
         }
         break;
      }
      break;
   }
}

void 
FWModelContextMenuHandler::colorChangeRequested(Color_t color)
{
   for(std::set<FWModelId>::const_iterator it =m_selectionManager->selected().begin(),
       itEnd = m_selectionManager->selected().end();
       it != itEnd;
       ++it) {
      FWDisplayProperties changeProperties = it->item()->modelInfo(it->index()).displayProperties();
      changeProperties.setColor(color);
      it->item()->setDisplayProperties(it->index(), changeProperties);
   }
}

void 
FWModelContextMenuHandler::addViewEntry(const char* iEntryName, int iEntryIndex)
{
   if(!m_viewSeperator) { 	 
      m_modelPopup->AddSeparator(m_afterViewSeperator); 	 
      m_viewSeperator=dynamic_cast<TGMenuEntry*>(m_modelPopup->GetListOfEntries()->Before(m_afterViewSeperator));
      assert(0!=m_viewSeperator); 	 
   }
   if(static_cast<int>(m_nViewEntries) > iEntryIndex) {
      m_modelPopup->GetEntry(iEntryIndex+kViewOptionsMO)->GetLabel()->SetString(iEntryName);
      m_modelPopup->EnableEntry(iEntryIndex+kViewOptionsMO);
   } else {
      assert(static_cast<int>(m_nViewEntries) == iEntryIndex);
      m_modelPopup->AddEntry(iEntryName,kViewOptionsMO+iEntryIndex,0,0,m_viewSeperator);
      ++m_nViewEntries;
   }
}

//
// const member functions
//
void 
FWModelContextMenuHandler::showSelectedModelContext(Int_t iX, Int_t iY, FWViewContextMenuHandlerBase* iHandler) const
{
   m_viewHander=iHandler;
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
      //add the detail view entries
      std::vector<std::string> viewChoices = m_detailViewManager->detailViewsFor(*(m_selectionManager->selected().begin()));
      if(viewChoices.size()>0) {
         if(m_nDetailViewEntries < viewChoices.size()) {
            for(unsigned int index = m_nDetailViewEntries;
                index != viewChoices.size();
                ++index) {
               m_modelPopup->AddEntry(kOpenDetailView,kOpenDetailViewMO+index,0,0,m_seperator);
            }
            m_nDetailViewEntries=viewChoices.size();
         }
         const std::string kStart("Open ");
         const std::string kEnd(" Detail View ...");
         for(unsigned int index=0; index != viewChoices.size(); ++index) {
            m_modelPopup->GetEntry(index+kOpenDetailViewMO)->GetLabel()->SetString((kStart+viewChoices[index]+kEnd).c_str());
            m_modelPopup->EnableEntry(index+kOpenDetailViewMO);
         }
         for(unsigned int i =viewChoices.size(); i <m_nDetailViewEntries; ++i) {
            m_modelPopup->HideEntry(kOpenDetailViewMO+i);
         }
         
      } else {
         for(unsigned int i =0; i <m_nDetailViewEntries; ++i) {
            m_modelPopup->HideEntry(kOpenDetailViewMO+i);
         }
      }
   } else {
      for(unsigned int i =0; i <m_nDetailViewEntries; ++i) {
         m_modelPopup->HideEntry(kOpenDetailViewMO+i);
      }
   }
   //add necessary entries from the view
   m_modelPopup->DeleteEntry(m_viewSeperator);
   m_viewSeperator=0;

   for(unsigned int i=0; i<m_nViewEntries; ++i) {
      m_modelPopup->HideEntry(kViewOptionsMO+i);
   }
   if(m_viewHander) {
      m_viewHander->addTo(const_cast<FWModelContextMenuHandler&>(*this));
   }
   
   m_x=iX;
   m_y=iY;
   m_modelPopup->PlaceMenu(iX,iY,false,true);
}

void 
FWModelContextMenuHandler::createModelContext() const
{
   if(0==m_modelPopup) {
      m_modelPopup = new FWPopupMenu();
      
      m_modelPopup->AddEntry("Set Visible",kSetVisibleMO);
      m_modelPopup->AddEntry("Set Color ...",kSetColorMO);
      m_modelPopup->AddEntry(kOpenDetailView,kOpenDetailViewMO);
      m_nDetailViewEntries=1;
      m_modelPopup->AddSeparator();
      m_seperator = dynamic_cast<TGMenuEntry*>(m_modelPopup->GetListOfEntries()->Last());
      assert(0!=m_seperator);
      m_modelPopup->AddEntry("Open Object Controller ...",kOpenObjectControllerMO);
      m_afterViewSeperator = dynamic_cast<TGMenuEntry*>(m_modelPopup->GetListOfEntries()->Last());
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
      std::vector<Color_t> colors;
      m_colorManager->fillLimitedColors(colors);
      
      m_colorPopup = new FWColorPopup(gClient->GetDefaultRoot(), colors.front());
      m_colorPopup->InitContent("", colors);
      m_colorPopup->Connect("ColorSelected(Color_t)","FWModelContextMenuHandler", const_cast<FWModelContextMenuHandler*>(this), "colorChangeRequested(Color_t)");
   }
}

//
// static member functions
//

ClassImp(FWModelContextMenuHandler)
