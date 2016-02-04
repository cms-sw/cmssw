#ifndef Fireworks_Core_FWModelContextMenuHandler_h
#define Fireworks_Core_FWModelContextMenuHandler_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelContextMenuHandler
// 
/**\class FWModelContextMenuHandler FWModelContextMenuHandler.h Fireworks/Core/interface/FWModelContextMenuHandler.h

 Description: Controls the context menus

 Usage:
    This file is used internally by the system

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Sep 22 13:26:02 CDT 2009
// $Id: FWModelContextMenuHandler.h,v 1.8 2011/03/25 18:02:46 amraktad Exp $
//

// system include files
#include "Rtypes.h"
#include "GuiTypes.h"

// user include files

// forward declarations
class TGPopupMenu;
class TGMenuEntry;
class FWSelectionManager;
class FWDetailViewManager;
class FWColorManager;
class FWColorPopup;
class FWGUIManager;
class FWViewContextMenuHandlerBase;

class FWModelContextMenuHandler
{

public:
   FWModelContextMenuHandler(FWSelectionManager*,
                             FWDetailViewManager*,
                             FWColorManager*,
                             FWGUIManager*);
   virtual ~FWModelContextMenuHandler();
   
   // ---------- const member functions ---------------------
   ///NOTE: iX and iY are in global coordinates
   void showSelectedModelContext(Int_t iX, Int_t iY, FWViewContextMenuHandlerBase*) const;
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void chosenItem(Int_t);
   void colorChangeRequested(Color_t);
   
   void addViewEntry(const char*, int, bool enabled= true);
   ClassDef(FWModelContextMenuHandler,0);
private:
   FWModelContextMenuHandler(const FWModelContextMenuHandler&); // stop default
   
   const FWModelContextMenuHandler& operator=(const FWModelContextMenuHandler&); // stop default
   
   void createModelContext() const;
   void createColorPopup() const;
   // ---------- member data --------------------------------
   mutable TGPopupMenu* m_modelPopup;
   mutable FWColorPopup* m_colorPopup;
   FWSelectionManager* m_selectionManager;
   FWDetailViewManager* m_detailViewManager;
   FWColorManager* m_colorManager;
   FWGUIManager* m_guiManager;
   mutable TGMenuEntry* m_seperator;
   mutable TGMenuEntry* m_viewSeperator;
   mutable TGMenuEntry* m_afterViewSeperator;
   mutable Int_t m_x;
   mutable Int_t m_y;
   mutable unsigned int m_nDetailViewEntries;
   mutable unsigned int m_nViewEntries;
   mutable FWViewContextMenuHandlerBase* m_viewHander;
};


#endif
