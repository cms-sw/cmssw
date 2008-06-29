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
// $Id: CSGAction.cc,v 1.1 2008/06/17 00:08:11 chrjones Exp $
//

// system include files
#include <TString.h>
#include <TGResourcePool.h>
#include <TQObject.h>
#include <KeySymbols.h>
#include <TGMenu.h>

// user include files
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/src/CSGConnector.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"


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
   m_frame = frame;
   m_name = name;
   m_toolTip = "";
   m_textButton = 0;
   m_picButton = 0;
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
   delete m_textButton;
   delete m_picButton;
   delete m_menu;
   delete m_connector;
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
  if (m_textButton != 0) m_textButton->SetToolTipText(tip.c_str(), m_frame->getDelay());
  if (m_picButton != 0) m_picButton->SetToolTipText(tip.c_str(), m_frame->getDelay());
  if (m_tools != 0) m_tools->fTipText = tip.c_str();
}

void CSGAction::createTextButton(TGCompositeFrame* p, TGLayoutHints* l, Int_t id, GContext_t norm, FontStruct_t font, UInt_t option) {
   if (m_textButton != 0) {
      delete m_textButton;
   }
   m_textButton = new TGTextButton(p, m_name.c_str(), id, norm, font, option);
   if (m_toolTip != "") m_textButton->SetToolTipText(m_toolTip.c_str(), m_frame->getDelay());
   p->AddFrame(m_textButton, l);
   TQObject::Connect(m_textButton, "Pressed()", "CSGConnector", m_connector, "handleTextButton()");
}

void CSGAction::createPictureButton(TGCompositeFrame* p, const TGPicture* pic, TGLayoutHints* l, Int_t id, GContext_t norm, UInt_t option) {
   if (m_picButton != 0) {
      delete m_picButton;
   }
   m_picButton = new TGPictureButton(p, pic, id, norm, option);
   if (m_toolTip != "") m_picButton->SetToolTipText(m_toolTip.c_str(), m_frame->getDelay());
   p->AddFrame(m_picButton, l);
   TQObject::Connect(m_picButton, "Pressed()", "CSGConnector", m_connector, "handlePictureButton()");
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
      scText = "<ctrl> "; 
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
  /*
   TString realName(m_name);
   TString subMenuName(m_name);
   TGPopupMenu *rootMenu;
   if (menubar->GetPopup(menu) == 0) {
      // Menu heading doesn't exist yet, so make it
      rootMenu = new TGPopupMenu(gClient->GetRoot());
      menubar->AddPopup(menu,rootMenu,new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
      TQObject::Connect(rootMenu, "Activated(Int_t)", "CSGConnector", m_connector, "handleMenu(Int_t)");
   }
   else {
      rootMenu = menubar->GetPopup(menu);
   }
   if (realName.Contains("->")) {
      // Have to add to submenu, so extract the names of submenu and of entry
      while (!(subMenuName.EndsWith("->")) && subMenuName.Length() > 0) {
         subMenuName.Resize(subMenuName.Length()-1);
      }
      subMenuName.Resize(subMenuName.Length()-2);
      while (!(realName.BeginsWith("->")) && realName.Length() > 0) {
         realName.Replace(0,1,0,0);
      }
      realName.Replace(0,2,0,0);
      TGPopupMenu *subMenu;
      if (rootMenu->GetEntry(subMenuName) != 0) {
         subMenu = rootMenu->GetEntry(subMenuName)->GetPopup();
         subMenu->AddEntry(realName, m_entry);
      }
      else {
         // Submenu doesn't exist yet, so make it
         subMenu = new TGPopupMenu(gClient->GetRoot());
         subMenu->AddEntry(realName, m_entry);
         rootMenu->AddPopup(subMenuName, subMenu);
         // Each entry in the entire bar has unique id, so send to same connector
         TQObject::Connect(subMenu, "Activated(Int_t)", "CSGConnector", m_connector, "handleMenu(Int_t)");
      }
      m_menu = subMenu;
   }
   else { 
      rootMenu->AddEntry(m_name.c_str(), m_entry);
      m_menu = rootMenu;
   }
  */
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
   toolbar->AddButton(m_frame,m_tools,5);
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
   while (current = (TGMenuEntry *)next()) {
      if (current == m_menu->GetEntry(m_entry)) {
         break;
      }
   }
   current = (TGMenuEntry *)next();
   m_menu->DeleteEntry(m_entry);
   m_menu->AddEntry(realName, m_entry, 0, 0, current);
   return widthChanged;
}  

TGTextButton* CSGAction::getTextButton() const {
   return m_textButton;
}

TGPictureButton* CSGAction::getPictureButton() const {
   return m_picButton;
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
  if (m_menu != 0) m_menu->EnableEntry(m_entry);
  if (m_textButton != 0) m_textButton->SetEnabled(kTRUE);
  if (m_picButton != 0) m_picButton->SetEnabled(kTRUE);
  if (m_toolBar != 0) m_toolBar->GetButton(m_entry)->SetEnabled(kTRUE);
  if (m_keycode != 0) {
    int id = m_frame->GetId();
    gVirtualX->GrabKey(id, m_keycode, m_modcode, kTRUE);
    gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyMod2Mask, kTRUE);
    gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyLockMask, kTRUE);
    gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyMod2Mask | kKeyLockMask, kTRUE);
  }
  m_enabled = kTRUE;
}

void CSGAction::disable() {
  if (m_menu != 0) m_menu->DisableEntry(m_entry);
  if (m_textButton != 0) m_textButton->SetEnabled(kFALSE);
  if (m_picButton != 0) m_picButton->SetEnabled(kFALSE);
  if (m_toolBar != 0) m_toolBar->GetButton(m_entry)->SetEnabled(kFALSE);
  if (m_keycode != 0) {
    int id = m_frame->GetId();
    gVirtualX->GrabKey(id, m_keycode, m_modcode, kFALSE);
    gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyMod2Mask, kFALSE);
    gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyLockMask, kFALSE);
    gVirtualX->GrabKey(id, m_keycode, m_modcode | kKeyMod2Mask | kKeyLockMask, kFALSE);
  }
  m_enabled = kFALSE;
}

Bool_t CSGAction::isEnabled() const {
  return m_enabled;
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

