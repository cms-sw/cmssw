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
// $Id: CmsShowModelPopup.h,v 1.17 2010/09/15 18:14:22 amraktad Exp $
//

// system include files
#include <set>
#include <vector>
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
class TGHSlider;
class FWDetailViewManager;
class FWSelectionManager;
class FWDialogBuilder;

class CmsShowModelPopup;

class CmsShowModelPopupDetailViewButtonAdapter {
public:
   CmsShowModelPopupDetailViewButtonAdapter(CmsShowModelPopup* iPopup,
                                            int iIndex):
   m_popup(iPopup),
   m_index(iIndex)
   {}
   void wasClicked();
private:
   CmsShowModelPopup* m_popup;
   int m_index;
};

class CmsShowModelPopup : public TGTransientFrame
{

public:
   friend class CmsShowModelPopupDetailViewButtonAdapter;
   
   CmsShowModelPopup(FWDetailViewManager*, FWSelectionManager*, 
                     const FWColorManager*, const TGWindow* p = 0, 
                     UInt_t w = 1, UInt_t h = 1);
   virtual ~CmsShowModelPopup();

   virtual void CloseWindow() { UnmapWindow(); }
   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void fillModelPopup(const FWSelectionManager& iSelMgr);
   void updateDisplay();
   void colorSetChanged();
   void disconnectAll();
   void changeModelColor(Color_t iColor);
   void changeModelOpacity(Int_t opacity = 100);
   void toggleModelVisible(Bool_t on = kTRUE);
   void openDetailedView();

private:
   CmsShowModelPopup(const CmsShowModelPopup&);    // stop default

   const CmsShowModelPopup& operator=(const CmsShowModelPopup&);    // stop default

   void clicked(int);
   
   // ---------- member data --------------------------------
   TGLabel* m_modelLabel;
   FWColorSelect* m_colorSelectWidget;
   TGCheckButton* m_isVisibleButton;
   std::vector<TGTextButton*> m_openDetailedViewButtons;
   std::vector<CmsShowModelPopupDetailViewButtonAdapter*> m_adapters;
   std::set<FWModelId> m_models;
   sigc::connection m_modelChangedConn;
   sigc::connection m_destroyedConn;
   sigc::connection m_changes;

   FWDetailViewManager* m_detailViewManager;
   const FWColorManager* m_colorManager;
   TGHSlider            *m_opacitySlider;
   FWDialogBuilder*     m_dialogBuilder;

   ClassDef(CmsShowModelPopup, 1);
};


#endif
