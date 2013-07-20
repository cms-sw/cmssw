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
// $Id: FWModelContextMenuHandler.cc,v 1.26 2013/01/25 19:36:33 wmtan Exp $
//

// system include files
#include <cassert>
#include "TGMenu.h"
#include "KeySymbols.h"

#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "TClass.h"

// user include files
#include "Fireworks/Core/src/FWModelContextMenuHandler.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/src/FWPopupMenu.cc"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWViewContextMenuHandlerBase.h"

//
// constants, enums and typedefs
//
enum MenuOptions {
   kSetVisibleMO,
   kSetColorMO,
   kPrint,
   kOpenDetailViewMO,
   kAfterOpenDetailViewMO,
   kOpenObjectControllerMO=100,
   kOpenCollectionControllerMO,
   kViewOptionsMO=1000,
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
m_seperator(0),
m_viewSeperator(0),
m_afterViewSeperator(0),
m_x(0),
m_y(0),
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
#include "TROOT.h"
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
      case kPrint:
      {
         FWModelId id = *(m_selectionManager->selected().begin());
         edm::TypeWithDict rtype(edm::TypeWithDict::byName(id.item()->modelType()->GetName()));
         edm::ObjectWithDict o(rtype, const_cast<void *>(id.item()->modelData(id.index())));

         // void* xx = &std::cout;
         //const std::vector<void*> j(1, xx);
         //edm::TypeMemberQuery inh =  edm::TypeMemberQuery::InheritedAlso;
         //edm::FunctionWithDict m = rtype.functionMemberByName("print",edm::TypeWithDict(edm::TypeWithDict::byName("void (std::ostream&)"), edm::TypeModifiers::Const), edm::TypeModifiers::NoMod , inh))
         //m.Invoke(o, 0, j);

         const char* cmd  = Form("FWGUIManager::OStream() << *(%s*)%p ;",  id.item()->modelType()->GetName(), (void*)id.item()->modelData(id.index()));
         //const char* cmd  = Form("*((std::ostream*)%p) << (%s*)%p ;", (void*)(&std::cout), id.item()->modelType()->GetName(), (void*)id.item()->modelData(id.index()));
         std::cout << cmd << std::endl;
         gROOT->ProcessLine(cmd);


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
FWModelContextMenuHandler::addViewEntry(const char* iEntryName, int iEntryIndex, bool enabled)
{
   if(!m_viewSeperator) { 	 
      m_modelPopup->AddSeparator(m_afterViewSeperator); 	 
      m_viewSeperator=dynamic_cast<TGMenuEntry*>(m_modelPopup->GetListOfEntries()->Before(m_afterViewSeperator));
      assert(0!=m_viewSeperator); 	 
   }
 
   if(static_cast<int>(m_nViewEntries) > iEntryIndex) {
      m_modelPopup->GetEntry(iEntryIndex+kViewOptionsMO)->GetLabel()->SetString(iEntryName);
      if(enabled)
         m_modelPopup->EnableEntry(iEntryIndex+kViewOptionsMO);
      else
         m_modelPopup->DisableEntry(iEntryIndex+kViewOptionsMO);

   } else {
      assert(static_cast<int>(m_nViewEntries) == iEntryIndex);
      m_modelPopup->AddEntry(iEntryName,kViewOptionsMO+iEntryIndex,0,0,m_viewSeperator);

      if (enabled)
         m_modelPopup->EnableEntry(kViewOptionsMO+iEntryIndex);
      else
         m_modelPopup->DisableEntry(kViewOptionsMO+iEntryIndex);

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


   if( m_selectionManager->selected().size()==1 ) {
      {
         edm::TypeWithDict rtype(edm::TypeWithDict::byName(id.item()->modelType()->GetName()));
         edm::ObjectWithDict o(rtype, const_cast<void *>(id.item()->modelData(id.index())));
         edm::TypeMemberQuery inh =  edm::TypeMemberQuery::InheritedAlso;
         if ( rtype.functionMemberByName("print",edm::TypeWithDict(edm::TypeWithDict::byName("void (std::ostream&)"), Long_t(kIsConstant)), 0, inh))
         {
            m_modelPopup->EnableEntry(kPrint);
            // std::cout <<  "Enable " <<std::endl;
         }
         else
         {           
            m_modelPopup->DisableEntry(kPrint);
            // printf("Disable print \n");
         }         
      }
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
      m_viewHander->addTo(const_cast<FWModelContextMenuHandler&>(*this), *(m_selectionManager->selected().begin()));
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
      m_modelPopup->AddEntry("Print ...",kPrint);
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
