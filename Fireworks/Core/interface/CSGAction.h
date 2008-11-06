#ifndef Fireworks_Core_CSGAction_h
#define Fireworks_Core_CSGAction_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGAction
//
/**\class CSGAction CSGAction.h Fireworks/Core/interface/CSGAction.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu May 29 18:15:56 CDT 2008
// $Id: CSGAction.h,v 1.6 2008/11/05 09:03:43 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include <sigc++/sigc++.h>
#include <TGFrame.h>
#include <TGButton.h>
#include <TGToolBar.h>

// user include files

// forward declarations
class CmsShowMainFrame;
class CSGConnector;
class TGMenuBar;
class TString;

class TGTextEntry;
class TGNumberEntryField;
class FWCustomIconsButton;

class CSGAction : public sigc::trackable {

public:
   CSGAction(CmsShowMainFrame *frame, const char *name);
   virtual ~CSGAction();

   // ---------- const member functions ---------------------
   const std::string& getName() const;
   const std::string& getToolTip() const;
   TString getSCCombo() const;
   TGTextEntry  *getTextEntry() const { return m_textEntry; }
   TGNumberEntryField *getNumberEntry() const { return m_numberEntry; }
   Int_t getKeycode() const;
   Int_t getModcode() const;
   TGPopupMenu *getMenu() const;
   int getMenuEntry() const;
   ToolBarData_t *getToolBarData() const;
   TGToolBar *getToolBar() const;
   virtual Bool_t isEnabled() const;

   // ---------- static member functions --------------------
   static TString keycodeToString(Int_t keycode);

   // ---------- member functions ---------------------------
   void setName(const std::string& name);
   void setToolTip(const std::string& tip);
   void createTextButton(TGCompositeFrame* p, TGLayoutHints* l = 0, Int_t id = -1, GContext_t norm = TGButton::GetDefaultGC()(), FontStruct_t font = TGTextButton::GetDefaultFontStruct(), UInt_t option = kRaisedFrame|kDoubleBorder);
   void createTextEntry(TGCompositeFrame* p, TGLayoutHints* l = 0, const char* text = 0, Int_t id = -1);
   void createNumberEntry(TGCompositeFrame* p,  bool intType, TGLayoutHints* l = 0, Int_t id = -1);
   void createPictureButton(TGCompositeFrame* p, const TGPicture* pic, TGLayoutHints* l = 0, Int_t id = -1, GContext_t norm = TGButton::GetDefaultGC()(), UInt_t option = kRaisedFrame|kDoubleBorder);
   FWCustomIconsButton* createCustomIconsButton(TGCompositeFrame* p,
                                const TGPicture* upPic,
                                const TGPicture* downPic,
                                const TGPicture* disabledPic,
                                TGLayoutHints* l = 0,
                                Int_t id = -1,
                                GContext_t norm = TGButton::GetDefaultGC()(),
                                UInt_t option = 0);
   void createShortcut(UInt_t key, const char *mod);
   void createMenuEntry(TGPopupMenu *menu);
   void createToolBarEntry(TGToolBar *toolbar, const char *filename);

   void enable();
   void disable();

   virtual void globalEnable();
   virtual void globalDisable();

   void addSCToMenu();
   Bool_t resizeMenuEntry();
   void activate(){ activated.emit(); }

   sigc::signal<void> activated;

private:
   CSGAction(const CSGAction&); // stop default

   const CSGAction& operator=(const CSGAction&); // stop default

   void enableImp();
   void disableImp();
   // ---------- member data --------------------------------
   CmsShowMainFrame *m_frame;
   std::string m_name;
   std::string m_toolTip;
   TString m_scCombo;
   std::vector<TGButton*> m_buttons;
   Int_t m_keycode;
   Int_t m_modcode;
   TGPopupMenu *m_menu;
   int m_entry;
   TGToolBar *m_toolBar;
   ToolBarData_t *m_tools;
   CSGConnector *m_connector;
   Bool_t m_enabled;
   Bool_t m_globalEnabled;
   TGTextEntry* m_textEntry;
   TGNumberEntryField* m_numberEntry;
};


#endif
