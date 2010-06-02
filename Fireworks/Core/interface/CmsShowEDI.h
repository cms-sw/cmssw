#ifndef Fireworks_Core_CmsShowEDI_h
#define Fireworks_Core_CmsShowEDI_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowEDI
//
/**\class CmsShowEDI CmsShowEDI.h Fireworks/Core/interface/CmsShowEDI.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Joshua Berger
//         Created:  Mon Jun 23 15:48:42 EDT 2008
// $Id: CmsShowEDI.h,v 1.12 2009/05/27 15:40:13 chrjones Exp $
//

// system include files
#include <sigc++/connection.h>
#include "GuiTypes.h"
#include "TGFrame.h"

// user include files
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWDataCategories.h"

// forward declarations
class FWSelectionManager;
class FWColorManager;
class FWEventItem;
class TGLabel;
class FWColorSelect;
class TGCheckButton;
class TGTextEntry;
class TGTextButton;
class TGTextView;
class TGComboBoxPopup;
class TGListBox;
class FWGUIValidatingTextEntry;
class FWExpressionValidator;
class TGTab;

class CmsShowEDI : public TGTransientFrame
{

public:
   CmsShowEDI(const TGWindow* p = 0, UInt_t w = 1, UInt_t h = 1, FWSelectionManager* selMgr = 0,
   FWColorManager* colorMgr=0);
   virtual ~CmsShowEDI();

   virtual void CloseWindow() { UnmapWindow(); }

   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------

   // ---------- member functions --------------------------
   void fillEDIFrame();
   void removeItem();
   //      void emptyEDIFrame();
   void updateDisplay();
   void updateFilter();
   void disconnectAll();
   void changeItemColor(Color_t color );
   void toggleItemVisible(Bool_t on = kTRUE);
   void runFilter();
   void runSelection();
   void selectAll();
   void moveToBack();
   void moveToFront();

   void show(FWDataCategories);
private:
   CmsShowEDI(const CmsShowEDI&);    // stop default

   const CmsShowEDI& operator=(const CmsShowEDI&);    // stop default

   // ---------- member data --------------------------------
   FWSelectionManager* m_selectionManager;
   TGTab* m_tabs;
   TGLabel* m_objectLabel;
   TGTextButton* m_removeButton;
   TGTextButton* m_frontButton;
   TGTextButton* m_backButton;
   FWColorSelect* m_colorSelectWidget;
   TGCheckButton* m_isVisibleButton;
   FWGUIValidatingTextEntry* m_filterExpressionEntry;
   FWGUIValidatingTextEntry* m_selectExpressionEntry;
   TGTextButton* m_filterButton;
   TGTextButton* m_selectButton;
   TGTextButton* m_selectAllButton;
   TGTextEntry* m_nameEntry;
   TGTextEntry* m_typeEntry;
   TGTextEntry* m_moduleEntry;
   TGTextEntry* m_instanceEntry;
   TGTextEntry* m_processEntry;
   FWEventItem* m_item;
   sigc::connection m_displayChangedConn;
   sigc::connection m_modelChangedConn;
   sigc::connection m_destroyedConn;
   TGTextView* m_filterError;
   TGTextView* m_selectError;
   FWExpressionValidator* m_validator;
   FWColorManager* m_colorManager;
};


#endif
