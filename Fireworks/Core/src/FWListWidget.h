// @(#)root/eve:$Id: FWListWidget.h,v 1.1 2008/07/08 20:08:04 chrjones Exp $
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
* Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#ifndef Fireworks_Core_src_FWListWidget
#define Fireworks_Core_src_FWListWidget

#include "TEveElement.h"
#include "TEveBrowser.h"

#include "TGListTree.h"

#include "TContextMenu.h"

class FWListWidget : public TGMainFrame
{
   FWListWidget(const FWListWidget&);            // Not implemented
   FWListWidget& operator=(const FWListWidget&); // Not implemented

protected:
   TGCompositeFrame *fFrame;
   //TGCompositeFrame *fLTFrame;

   TGCanvas         *fLTCanvas;
   TGListTree       *fListTree;

   TContextMenu     *fCtxMenu;

   Bool_t fSignalsConnected;

public:
   FWListWidget(const TGWindow* p=0, Int_t width=250, Int_t height=700);
   virtual ~FWListWidget();

   void ConnectSignals();
   void DisconnectSignals();

   TGListTree*    GetListTree() const {
      return fListTree;
   }

   void ItemBelowMouse(TGListTreeItem *entry, UInt_t mask);
   void ItemClicked(TGListTreeItem *entry, Int_t btn, UInt_t mask, Int_t x, Int_t y);
   void ItemDblClicked(TGListTreeItem* item, Int_t btn);
   void ItemKeyPress(TGListTreeItem *entry, UInt_t keysym, UInt_t mask);

   ClassDef(FWListWidget, 0); // Composite GUI frame for parallel display of a TGListTree and TEveGedEditor.
};
#endif
