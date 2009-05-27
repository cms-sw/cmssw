#ifndef Fireworks_Core_CmsShowModelPopup_h
#define Fireworks_Core_CmsShowModelPopup_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowModelPopup
//
/**\class CmsShowModelPopup CmsShowModelPopup.h Fireworks/Core/interface/CmsShowModelPopup.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Fri Jun 27 11:23:31 EDT 2008
// $Id: CmsShowModelPopup.h,v 1.11 2009/05/20 16:33:39 amraktad Exp $
//

// system include files
#include <set>
#include <sigc++/connection.h>
#include "GuiTypes.h"
#include "TGFrame.h"

// user include files
#include "Fireworks/Core/interface/FWModelChangeSignal.h"

// forward declarations
class FWEventItem;
class FWSelectionManager;
class FWColorManager;
//class FWModelId;
class FWColorSelect;
class TGLabel;
class TGTextButton;
class TGTextButton;
class FWDetailViewManager;
class FWSelectionManager;

class CmsShowModelPopup : public TGTransientFrame
{

public:
   CmsShowModelPopup(FWDetailViewManager*, FWSelectionManager*, const FWColorManager*, const TGWindow* p = 0, UInt_t w = 1, UInt_t h = 1);
   virtual ~CmsShowModelPopup();

  virtual void CloseWindow() { UnmapWindow(); }
   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void fillModelPopup(const FWSelectionManager& iSelMgr);
   void updateDisplay();
   void disconnectAll();
   void changeModelColor(Color_t iColor);
   void toggleModelVisible(Bool_t on = kTRUE);
   void openDetailedView();

private:
   CmsShowModelPopup(const CmsShowModelPopup&);    // stop default

   const CmsShowModelPopup& operator=(const CmsShowModelPopup&);    // stop default

   // ---------- member data --------------------------------
   TGLabel* m_modelLabel;
   FWColorSelect* m_colorSelectWidget;
   TGCheckButton* m_isVisibleButton;
   TGTextButton* m_openDetailedViewButton;
   std::set<FWModelId> m_models;
   sigc::connection m_modelChangedConn;
   sigc::connection m_destroyedConn;
   sigc::connection m_changes;

   FWDetailViewManager* m_detailViewManager;
   const FWColorManager* m_colorManager;
};


#endif
