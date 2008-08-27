// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGUIValidatingTextEntry
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Aug 22 18:13:39 EDT 2008
// $Id: FWGUIValidatingTextEntry.cc,v 1.1 2008/08/24 00:19:12 chrjones Exp $
//

// system include files
#include <iostream>
#include "TGComboBox.h"

// user include files
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "Fireworks/Core/src/FWExpressionValidator.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWGUIValidatingTextEntry::FWGUIValidatingTextEntry(const TGWindow *parent , const char *text , Int_t id ):
TGTextEntry(parent,text,id),
m_validator(0)
{
   m_popup = new TGComboBoxPopup(fClient->GetDefaultRoot(), 100, 100, kVerticalFrame);
   m_list = new TGListBox(m_popup, 1/*widget id*/, kChildFrame);
   m_list->Resize(100,100);
   m_list->Associate(this);
   m_list->GetScrollBar()->GrabPointer(kFALSE);
   m_popup->AddFrame(m_list, new TGLayoutHints(kLHintsExpandX| kLHintsExpandY));
   m_popup->MapSubwindows();
   m_popup->Resize(m_popup->GetDefaultSize());
   m_list->GetContainer()->AddInput(kButtonPressMask | kButtonReleaseMask | kPointerMotionMask);
   m_list->SetEditDisabled(kEditDisable);
   m_list->GetContainer()->SetEditDisabled(kEditDisable);
   Connect("TabPressed()", "FWGUIValidatingTextEntry", this, "showOptions()");

}

// FWGUIValidatingTextEntry::FWGUIValidatingTextEntry(const FWGUIValidatingTextEntry& rhs)
// {
//    // do actual copying here;
// }

FWGUIValidatingTextEntry::~FWGUIValidatingTextEntry()
{
}

//
// assignment operators
//
// const FWGUIValidatingTextEntry& FWGUIValidatingTextEntry::operator=(const FWGUIValidatingTextEntry& rhs)
// {
//   //An exception safe implementation is
//   FWGUIValidatingTextEntry temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWGUIValidatingTextEntry::setValidator(FWValidatorBase* iValidator)
{
   m_validator = iValidator;
}


Bool_t 
FWGUIValidatingTextEntry::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   //STOLEN FROM TGComboBox.cxx
   TGLBEntry *e;
   
   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_LISTBOX:
               InsertText(m_options[m_list->GetSelected()].second.c_str(), GetCursorPosition());
               m_popup->EndPopup();
               fClient->NeedRedraw(this);
               break;
         }
         break;
         
      default:
         break;
   }
   return kTRUE;
   
}

void 
FWGUIValidatingTextEntry::showOptions() {

   if(0!=m_validator) {
      const char* text = GetText();
      std::string subText(text,text+GetCursorPosition());
      //std::cout <<subText<<std::endl;
      
      typedef std::vector<std::pair<boost::shared_ptr<std::string>, std::string> > Options;
      m_validator->fillOptions(text, text+GetCursorPosition(), m_options);
      if(m_options.empty()) { return;}
      if(m_options.size()==1) {
         long pos = GetCursorPosition();
         InsertText(m_options.front().second.c_str(), GetCursorPosition());
         SetCursorPosition(pos + m_options.front().second.size());
         return;
      }
      m_list->RemoveAll();
      int index = 0;
      for(Options::iterator it = m_options.begin(), itEnd = m_options.end();
          it != itEnd; ++it,++index) {
         m_list->AddEntry(it->first->c_str(),index);
      }
      {
         unsigned int h = m_list->GetNumberOfEntries()*
         m_list->GetItemVsize();
         if(h && (h<100)) {
            m_list->Resize(m_list->GetWidth(),h);
         } else {
            m_list->Resize(m_list->GetWidth(),100);
         }
      }
      
      int ax,ay;
      Window_t wdummy;
      gVirtualX->TranslateCoordinates(GetId(), m_popup->GetParent()->GetId(),
                                      0, GetHeight(), ax, ay, wdummy);
      
      m_popup->PlacePopup(ax, ay, 
                          GetWidth()-2, m_popup->GetDefaultHeight());
   }
}
//
// const member functions
//

//
// static member functions
//
