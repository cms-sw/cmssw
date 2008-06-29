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
// $Id$
//

// system include files
#include <sigc++/connection.h>
#include "GuiTypes.h"
#include "TGFrame.h"

// user include files
#include "Fireworks/Core/interface/FWModelChangeSignal.h"

// forward declarations
class FWEventItem;
class FWListModel;
class FWColorSelect;
class TGLabel;
class TGTextButton;
class TGTextButton;

class CmsShowModelPopup : public TGMainFrame
{

   public:
      CmsShowModelPopup(const TGWindow* p = 0, UInt_t w = 1, UInt_t h = 1);
      virtual ~CmsShowModelPopup();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void fillModelPopup(FWListModel* iModel = 0);
      void removeModel();
      void updateDisplay();
      void disconnectAll();
      void changeModelColor(Pixel_t pixel = 0x000000);
      void toggleModelVisible(Bool_t on = kTRUE);

   private:
      CmsShowModelPopup(const CmsShowModelPopup&); // stop default

      const CmsShowModelPopup& operator=(const CmsShowModelPopup&); // stop default

      // ---------- member data --------------------------------
      TGLabel* m_modelLabel;
      TGTextButton* m_removeButton;
      FWColorSelect* m_colorSelectWidget;
      TGCheckButton* m_isVisibleButton;
      FWListModel* m_model;
      FWEventItem* m_item;
      //      sigc::connection m_displayChangedConn;
      sigc::connection m_modelChangedConn;
      sigc::connection m_destroyedConn;

};


#endif
