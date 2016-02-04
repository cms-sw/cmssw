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
// $Id: FWGUIValidatingTextEntry.cc,v 1.9 2011/07/20 20:17:54 amraktad Exp $
//

// system include files
#include <iostream>
#include "TGComboBox.h"
#include "KeySymbols.h"
#include "TTimer.h"
#include "TGWindow.h"

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
FWGUIValidatingTextEntry::FWGUIValidatingTextEntry(const TGWindow *parent, const char *text, Int_t id ) :
   TGTextEntry(parent,text,id),
   m_popup(0),
   m_list(0),
   m_validator(0),
   m_listHeight(100)
{
   m_popup = new TGComboBoxPopup(fClient->GetDefaultRoot(), 100, 100, kVerticalFrame);
   m_list = new TGListBox(m_popup, 1 /*widget id*/, kChildFrame);
   m_list->Resize(100,m_listHeight);
   m_list->Associate(this);
   m_list->GetScrollBar()->GrabPointer(kFALSE);
   m_popup->AddFrame(m_list, new TGLayoutHints(kLHintsExpandX| kLHintsExpandY));
   m_popup->MapSubwindows();
   m_popup->Resize(m_popup->GetDefaultSize());
   m_list->GetContainer()->AddInput(kButtonPressMask | kButtonReleaseMask | kPointerMotionMask);
   m_list->SetEditDisabled(kEditDisable);
   m_list->GetContainer()->Connect("KeyPressed(TGFrame*,UInt_t,UInt_t)",
                                   "FWGUIValidatingTextEntry", this,
                                   "keyPressedInPopup(TGFrame*,UInt_t,UInt_t)");
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
   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_LISTBOX:
               RequestFocus();
               insertTextOption(m_options[m_list->GetSelected()].second);
               hideOptions();
               break;
         }
         break;

      default:
         break;
   }
   return kTRUE;

}

void
FWGUIValidatingTextEntry::keyPressedInPopup(TGFrame* f, UInt_t keysym, UInt_t mask)
{
   switch(keysym) {
      case kKey_Tab:
      case kKey_Escape:
         RequestFocus();
         hideOptions();
         break;
      case kKey_Return:
         RequestFocus();
         //NOTE: If chosen from the keyboard, m_list->GetSelected() does not work, however
         // m_list->GetSelectedEntries does work

         //AMT NOTE: TGListEntry does not select entry on key return event, it has to be selected here.
         //          Code stolen from TGComboBox::KeyPressed

         const TGLBEntry* entry = dynamic_cast<TGLBEntry*> (f);
         if (entry)
         {
            insertTextOption(m_options[entry->EntryId()].second);
            m_list->Selected(entry->EntryId());
         }
         hideOptions();
         break;
   }
}

namespace {
   class ChangeFocusTimer : public TTimer {
public:
      ChangeFocusTimer(TGWindow* iWindow) :
         TTimer(100),
         m_window(iWindow) {
      }
      virtual Bool_t Notify() {
         TurnOff();
         m_window->RequestFocus();
         return kTRUE;
      }
private:
      TGWindow* m_window;
   };
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
         insertTextOption(m_options.front().second);
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
         if(h && (h<m_listHeight)) {
            m_list->Resize(m_list->GetWidth(),h);
         } else {
            m_list->Resize(m_list->GetWidth(),m_listHeight);
         }
      }
      m_list->Select(0,kTRUE);

      int ax,ay;
      Window_t wdummy;
      gVirtualX->TranslateCoordinates(GetId(), m_popup->GetParent()->GetId(),
                                      0, GetHeight(), ax, ay, wdummy);

      //Wait to change focus for when the popup has already openned
      std::auto_ptr<TTimer> timer( new ChangeFocusTimer(m_list->GetContainer()) );
      timer->TurnOn();
      //NOTE: this call has its own internal GUI event loop and will not return
      // until the popup has been shut down
      m_popup->PlacePopup(ax, ay,
                          GetWidth()-2, m_popup->GetDefaultHeight());
   }
}

void
FWGUIValidatingTextEntry::hideOptions() {
   m_popup->EndPopup();
   fClient->NeedRedraw(this);
}

void
FWGUIValidatingTextEntry::insertTextOption(const std::string& iOption)
{
   long pos = GetCursorPosition();
   InsertText(iOption.c_str(), pos);
   SetCursorPosition(pos + iOption.size());
}

//
// const member functions
//

//
// static member functions
//
