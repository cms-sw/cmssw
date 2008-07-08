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
// $Id: CSGAction.h,v 1.1 2008/06/17 00:08:11 chrjones Exp $
//

// system include files
#include <string>
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

class CSGAction : public sigc::trackable {

public:
   CSGAction(CmsShowMainFrame *frame, const char *name);
   virtual ~CSGAction();
   
   // ---------- const member functions ---------------------
   const std::string& getName() const;
   const std::string& getToolTip() const;
   TString getSCCombo() const;
   TGTextButton *getTextButton() const;
   TGTextEntry  *getTextEntry() const { return m_textEntry; }
   TGPictureButton *getPictureButton() const;
   Int_t getKeycode() const;
   Int_t getModcode() const;
   TGPopupMenu *getMenu() const;
   int getMenuEntry() const;
   ToolBarData_t *getToolBarData() const;
   TGToolBar *getToolBar() const;
   Bool_t isEnabled() const;
   
   // ---------- static member functions --------------------
   static TString keycodeToString(Int_t keycode);
   
   // ---------- member functions ---------------------------
   void setName(const std::string& name);
   void setToolTip(const std::string& tip);
   void createTextButton(TGCompositeFrame* p, TGLayoutHints* l = 0, Int_t id = -1, GContext_t norm = TGButton::GetDefaultGC()(), FontStruct_t font = TGTextButton::GetDefaultFontStruct(), UInt_t option = kRaisedFrame|kDoubleBorder);
   void createTextEntry(TGCompositeFrame* p, TGLayoutHints* l = 0, const char* text = 0, Int_t id = -1);
   void createPictureButton(TGCompositeFrame* p, const TGPicture* pic, TGLayoutHints* l = 0, Int_t id = -1, GContext_t norm = TGButton::GetDefaultGC()(), UInt_t option = kRaisedFrame|kDoubleBorder);
   void createShortcut(UInt_t key, const char *mod);
   void createMenuEntry(TGPopupMenu *menu);
   void createToolBarEntry(TGToolBar *toolbar, const char *filename);

   void enable();
   void disable();
   
   void addSCToMenu();
   Bool_t resizeMenuEntry();
   void activate(){ activated.emit(); }
	
   sigc::signal<void> activated;
   
private:
   CSGAction(const CSGAction&); // stop default
   
   const CSGAction& operator=(const CSGAction&); // stop default
   
   // ---------- member data --------------------------------
   CmsShowMainFrame *m_frame;
   std::string m_name;
   std::string m_toolTip;
   TString m_scCombo;
   TGTextButton *m_textButton;
   TGPictureButton *m_picButton;
   Int_t m_keycode;
   Int_t m_modcode;
   TGPopupMenu *m_menu;
   int m_entry;
   TGToolBar *m_toolBar;
   ToolBarData_t *m_tools;
   CSGConnector *m_connector;
   Bool_t m_enabled;
   TGTextEntry* m_textEntry;
};


#endif
