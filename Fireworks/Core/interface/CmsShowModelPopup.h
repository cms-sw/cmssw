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
// $Id: CmsShowModelPopup.h,v 1.8 2009/01/23 21:35:40 amraktad Exp $
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

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void fillModelPopup(const FWSelectionManager& iSelMgr);
   void updateDisplay();
   void disconnectAll();
   void changeModelColor(Pixel_t pixel = 0x000000);
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
};


#endif
