// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGAction
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu May 29 20:58:11 CDT 2008
// $Id: CSGAction.cc,v 1.19 2009/01/06 17:16:00 amraktad Exp $
//

// system include files
#include <TString.h>
#include <TGResourcePool.h>
#include <TQObject.h>
#include <KeySymbols.h>
#include <TGMenu.h>
#include "TGLabel.h"

// user include files
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/src/CSGConnector.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CSGAction::CSGAction(CmsShowMainFrame *frame, const char *name) {
   m_enabled = kTRUE;
   m_globalEnabled = kTRUE;
   m_frame = frame;
   m_name = name;
   m_toolTip = "";
   m_menu = 0;
   m_toolBar = 0;
   m_tools = 0;
   m_connector = new CSGConnector(this, m_frame);
   m_frame->addToActionMap(this);
   m_entry = m_frame->getListOfActions().size();
   m_keycode = 0;
   m_modcode = 0;
}
// CSGAction::CSGAction(const CSGAction& rhs)
// {
//    // do actual copying here;
// }

CSGAction::~CSGAction()
{
   delete m_connector;
   //Don't delete GUI parts since they are owned by their GUI parent
}

//
// assignment operators
//
// const CSGAction& CSGAction::operator=(const CSGAction& rhs)
// {
//   //An exception safe implementation is
//   CSGAction temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
const std::string& CSGAction::getName() const {
   return m_name;
}

const std::string& CSGAction::getToolTip() const {
  return m_toolTip;
}

TString CSGAction::getSCCombo() const {
   return m_scCombo;
}

void CSGAction::setName(const std::string& name) {
   // Does not update menu yet
   m_name = name;
}

void CSGAction::setToolTip(const std::string& tip) {
   m_toolTip = tip;
   for(std::vector<TGButton*>::iterator it = m_buttons.begin(), itEnd = m_buttons.end();
       it != itEnd;
       ++it) {
      (*it)->SetToolTipText(tip.c_str(), m_frame->getToolTipDelay());
   }
  if (m_tools != 0) m_tools->fTipText = tip.c_str();
}

void CSGAction::createTextButton(TGCompositeFrame* p, TGLayoutHints* l, Int_t id, GContext_t norm, FontStruct_t font, UInt_t option) {
   TGTextButton* textButton = new TGTextButton(p, m_name.c_str(), id, norm, font, option);
   if (m_toolTip != "") textButton->SetToolTipText(m_toolTip.c_str(), m_frame->getToolTipDelay());
   p->AddFrame(textButton, l);
   TQObject::Connect(textButton, "Clicked()", "CSGAction", this, "activate()");
   m_buttons.push_back(textButton);
}

void CSGAction::createPictureButton(TGCompositeFrame* p, const TGPicture* pic, TGLayoutHints* l, Int_t id, GContext_t norm, UInt_t option) {
   TGPictureButton* picButton = new TGPictureButton(p, pic, id, norm, option);
   if (m_toolTip != "") picButton->SetToolTipText(m_toolTip.c_str(), m_frame->getToolTipDelay());
   p->AddFrame(picButton, l);
   TQObject::Connect(picButton, "Clicked()", "CSGAction", this, "activate()");
   m_buttons.push_back(picButton);
}

FWCustomIconsButton*
CSGAction::createCustomIconsButton(TGCompositeFrame* p,
                                   const TGPicture* upPic,
                                   const TGPicture* downPic,
                                   const TGPicture* disabledPic,
                                   TGLayoutHints* l,
                                   Int_t id ,
                                   GContext_t norm,
                                   UInt_t option)
{
   FWCustomIconsButton* picButton = new FWCustomIconsButton(p, upPic, downPic, disabledPic, id, norm, option);
   if (m_toolTip != "") picButton->SetToolTipText(m_toolTip.c_str(), m_frame->getToolTipDelay());
   p->AddFrame(picButton, l);
   TQObject::Connect(picButton, "Clicked()", "CSGAction", this, "activate()");
   m_buttons.push_back(picButton);
   return picButton;
}


void CSGAction::createShortcut(UInt_t key, const char *mod) {
   Int_t keycode = gVirtualX->KeysymToKeycode((int)key);
   Int_t modcode;

   TString scText;
   if (strcmp(mod, "CTRL") == 0) {
      modcode = kKeyControlMask;
      scText = "<ctrl> ";
   }
   else if (strcmp(mod, "CTRL+SHIFT") == 0) {
      modcode = kKeyControlMask | kKeyShiftMask;
      scText = "<ctrl> <shift> ";
   }
   else {
      // Default to ALT for now
      modcode = kKeyMod1Mask;
      scText = "<alt> ";
   }
   scText += keycodeToString(keycode);
   m_scCombo = scText;

   int id = m_frame->GetId();
   gVirtualX->GrabKey(id, keycode, modcode, kTRUE);
   gVirtualX->GrabKey(id, keycode, modcode | kKeyMod2Mask, kTRUE);
   gVirtualX->GrabKey(id, keycode, modcode | kKeyLockMask, kTRUE);
   gVirtualX->GrabKey(id, keycode, modcode | kKeyMod2Mask | kKeyLockMask, kTRUE);

   m_keycode = keycode;
   m_modcode = modcode;
   if (m_menu != 0) addSCToMenu();
}

void CSGAction::createMenuEntry(TGPopupMenu *menu) {
  m_menu = menu;
  if (!(menu->HasConnection("Activated(Int_t)"))) TQObject::Connect(menu, "Activated(Int_t)", "CSGConnector", m_connector, "handleMenu(Int_t)");
  menu->AddEntry(m_name.c_str(), m_entry);
  if (m_keycode != 0) addSCToMenu();
}

void CSGAction::createToolBarEntry(TGToolBar *toolbar, const char *filename) {
   m_toolBar = toolbar;
   m_tools = new ToolBarData_t();
   m_tools->fPixmap = filename;
   m_tools->fStayDown = kFALSE;
   m_tools->fId = m_entry;
   if(m_toolTip.size()) {m_tools->fTipText = m_toolTip.c_str();}
   TGButton* newButton = toolbar->AddButton(m_frame,m_tools,5);
   newButton->SetBackgroundColor(toolbar->GetBackground());
   newButton->ChangeOptions(0);

   int size = toolbar->GetList()->GetSize();
   if (size == 1) {
      // First button in tool bar, so connect the bar
      TQObject::Connect(toolbar, "Clicked(Int_t)", "CSGConnector", m_connector, "handleToolBar(Int_t)");
   }
}

void CSGAction::addSCToMenu() {
   Bool_t widthChanged = resizeMenuEntry();
   if (widthChanged) m_frame->resizeMenu(m_menu);
}

Bool_t CSGAction::resizeMenuEntry() {
   FontStruct_t font = gClient->GetResourcePool()->GetMenuHiliteFont()->GetFontStruct();
   Bool_t widthChanged = kTRUE;
   UInt_t width = m_menu->GetWidth();
   TString realName(m_name);
   if (realName.Contains("->")) {
      // Should make function to do this and store in member data...
      while (!(realName.BeginsWith("->")) && realName.Length() > 0) {
         realName.Replace(0,1,0,0);
      }
      realName.Replace(0,2,0,0);
   }
   TString scText = m_scCombo;
   while (gVirtualX->TextWidth(font, realName.Data(), realName.Length()) + gVirtualX->TextWidth(font, scText.Data(), scText.Length()) + 53 < (Int_t)width) {
      widthChanged = kFALSE;
      realName += " ";
   }
   realName += "     ";
   realName += scText;
   TIter next(m_menu->GetListOfEntries());
   TGMenuEntry *current;
   while (0 != (current = (TGMenuEntry *)next())) {
      if (current == m_menu->GetEntry(m_entry)) {
         break;
      }
   }
   current = (TGMenuEntry *)next();
   m_menu->DeleteEntry(m_entry);
   m_menu->AddEntry(realName, m_entry, 0, 0, current);
   return widthChanged;
}

TGPopupMenu* CSGAction::getMenu() const {
   return m_menu;
}

int CSGAction::getMenuEntry() const {
   return m_entry;
}

Int_t CSGAction::getKeycode() const {
   return m_keycode;
}

Int_t CSGAction::getModcode() const {
   return m_modcode;
}

ToolBarData_t* CSGAction::getToolBarData() const {
   return m_tools;
}

TGToolBar* CSGAction::getToolBar() const {
  return m_toolBar;
}

void CSGAction::enable() {
  m_enabled = kTRUE;
   enableImp();
}

void CSGAction::disable() {
  m_enabled = kFALSE;
   disableImp();
}

void
CSGAction::globalEnable()
{
   m_globalEnabled=true;
   enableImp();
}

void
CSGAction::globalDisable()
{
   m_globalEnabled=false;
   disableImp();
}


Bool_t CSGAction::isEnabled() const {
  return m_enabled && m_globalEnabled;
}

void CSGAction::enableImp() {
   if(isEnabled()) {
      if (m_menu != 0) m_menu->EnableEntry(m_entry);
      for(std::vector<TGButton*>::iterator it = m_buttons.begin(), itEnd = m_buttons.end();
          it != itEnd;
          ++it) {
         (*it)->SetEnabled(kTRUE);
      }

      if (m_toolBar != 0) m_toolBar->GetButton(m_entry)->SetEnabled(kTRUE);
      if (m_keycode != 0) {
         int id = m_frame->GetId();
         gVirtualX->GrabKey(id, m_keycode, m_modcode, kTRUE);
         gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyMod2Mask, kTRUE);
         gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyLockMask, kTRUE);
         gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyMod2Mask | kKeyLockMask, kTRUE);
      }
   }
}

void CSGAction::disableImp() {
   if(!isEnabled()) {
      if (m_menu != 0) m_menu->DisableEntry(m_entry);
      for(std::vector<TGButton*>::iterator it = m_buttons.begin(), itEnd = m_buttons.end();
          it != itEnd;
          ++it) {
         (*it)->SetEnabled(kFALSE);
      }
      if (m_toolBar != 0) m_toolBar->GetButton(m_entry)->SetEnabled(kFALSE);
      if (m_keycode != 0) {
         int id = m_frame->GetId();
         gVirtualX->GrabKey(id, m_keycode, m_modcode, kFALSE);
         gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyMod2Mask, kFALSE);
         gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyLockMask, kFALSE);
         gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyMod2Mask | kKeyLockMask, kFALSE);
      }
   }
}


//
// static member functions
//

TString
CSGAction::keycodeToString(Int_t keycode) {
   int i;
   char letter;
   TString rep;
   for (i = kKey_a; i < kKey_a + 26; i++) {
      if (gVirtualX->KeysymToKeycode(i) == keycode) {
         letter = (char)(i - kKey_a + 'a');
         rep = TString(letter);
         return rep;
      }
   }
   for (i = kKey_A; i < kKey_A + 26; i++) {
      if(gVirtualX->KeysymToKeycode(i) == keycode) {
         letter = (char)(i - kKey_A + 'a');
         rep = TString(letter);
         return rep;
      }
   }
   if (keycode == gVirtualX->KeysymToKeycode(kKey_Left)) {
      rep = TString("<-");
      return rep;
   }
   if (keycode == gVirtualX->KeysymToKeycode(kKey_Right)) {
      rep = TString("->");
      return rep;
   }
   rep = TString("");
   return rep;
}

