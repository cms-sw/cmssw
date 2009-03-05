// @(#)root/eve:$Id: FWListWidget.cc,v 1.4 2009/01/23 21:35:43 amraktad Exp $
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
* Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "Fireworks/Core/src/FWListWidget.h"

#include "TEveUtil.h"
#include "TEveElement.h"
#include "TEveManager.h"
#include "TEveSelection.h"


#include <Riostream.h>

#include "TClass.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TRint.h"
#include "TVirtualX.h"
#include "TEnv.h"

#include "TApplication.h"
#include "TFile.h"
#include "TClassMenuItem.h"

#include "TColor.h"

#include "TGCanvas.h"
#include "TGSplitter.h"
#include "TGStatusBar.h"
#include "TGMenu.h"
#include "TGPicture.h"
#include "TGToolBar.h"
#include "TGLabel.h"
#include "TGXYLayout.h"
#include "TGNumberEntry.h"
#include <KeySymbols.h>

#include "TGLSAViewer.h"
#include "TGLSAFrame.h"
#include "TGTab.h"


//==============================================================================
//==============================================================================
// TEveListTreeItem
//==============================================================================

//______________________________________________________________________________
//
// Special list-tree-item for Eve.
//
// Most state is picked directly from TEveElement, no need to store it
// locally nor to manage its consistency.
//
// Handles also selected/highlighted colors and, in the future,
// drag-n-drop.

ClassImp(TEveListTreeItem)

//______________________________________________________________________________
void TEveListTreeItem::NotSupported(const char* func) const
{
   // Warn about access to function members that should never be called.
   // TGListTree calls them in cases that are not used by Eve.

   Warning(Form("TEveListTreeItem::%s()", func), "not supported.");
}

//______________________________________________________________________________
Pixel_t TEveListTreeItem::GetActiveColor() const
{
   // Return highlight color corresponding to current state of TEveElement.

   switch (fElement->GetSelectedLevel())
   {
      case 1: return TColor::Number2Pixel(kBlue - 2);
      case 2: return TColor::Number2Pixel(kBlue - 6);
      case 3: return TColor::Number2Pixel(kCyan - 2);
      case 4: return TColor::Number2Pixel(kCyan - 6);
   }
   return TGFrame::GetDefaultSelectedBackground();
}

//______________________________________________________________________________
void TEveListTreeItem::Toggle()
{
   // Item's check-box state has been toggled ... forward to element's
   // render-state.

   fElement->SetRnrState(!IsChecked());
   fElement->ElementChanged(kTRUE, kTRUE);
}


//==============================================================================
//==============================================================================
// FWListWidget
//==============================================================================

//______________________________________________________________________________
//
// Composite GUI frame for parallel display of a TGListTree and TEveGedEditor.
//

ClassImp(FWListWidget)

//______________________________________________________________________________
FWListWidget::FWListWidget(const TGWindow* p, Int_t width, Int_t height) :
   TGMainFrame (p ? p : gClient->GetRoot(), width, height),
   fFrame      (0),
   //fLTFrame    (0),
   fListTree   (0),
   fCtxMenu    (0),
   fSignalsConnected (kFALSE)
{
   // Constructor.

   SetCleanup(kNoCleanup);

   fFrame = new TGCompositeFrame(this, width, height, kVerticalFrame);

   // List-tree
   //fLTFrame  = new TGCompositeFrame(fFrame, width, 3*height/7, kVerticalFrame);
   fLTCanvas = new TGCanvas(fFrame, 10, 10, kSunkenFrame | kDoubleBorder);
   fListTree = new TGListTree(fLTCanvas->GetViewPort(), 10, 10, kHorizontalFrame);
   fListTree->SetCanvas(fLTCanvas);
   fListTree->Associate(fFrame);
   fListTree->SetColorMode(TGListTree::EColorMarkupMode(TGListTree::kColorUnderline | TGListTree::kColorBox));
   fListTree->SetAutoCheckBoxPic(kFALSE);
   fListTree->SetUserControl(kTRUE);
   fLTCanvas->SetContainer(fListTree);
   fFrame->AddFrame(fLTCanvas, new TGLayoutHints
                              (kLHintsNormal | kLHintsExpandX | kLHintsExpandY, 1, 1, 1, 1));
   /*fFrame  ->AddFrame(fLTFrame, new TGLayoutHints
                      (kLHintsNormal | kLHintsExpandX | kLHintsExpandY));*/

   /*
      // Splitter
      fSplitter = new TGHSplitter(fFrame);
      fFrame->AddFrame(fSplitter, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 1,1,2,2));

      // Editor
      fFrame->SetEditDisabled(kEditEnable);
      fFrame->SetEditable();
      fEditor = new TEveGedEditor(0, width, 4*height/7);
      fEditor->SetGlobal(kFALSE);
      fEditor->ChangeOptions(fEditor->GetOptions() | kFixedHeight);
      fFrame->SetEditable(kEditDisable);
      fFrame->SetEditable(kFALSE);
      {
      TGFrameElement *el = 0;
      TIter next(fFrame->GetList());
      while ((el = (TGFrameElement *) next())) {
         if (el->fFrame == fEditor)
            if (el->fLayout) {
               el->fLayout->SetLayoutHints(kLHintsTop | kLHintsExpandX);
               el->fLayout->SetPadLeft(0); el->fLayout->SetPadRight(1);
               el->fLayout->SetPadTop(2);  el->fLayout->SetPadBottom(1);
               break;
            }
      }
      }
      fSplitter->SetFrame(fEditor, kFALSE);
    */
   AddFrame(fFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));

   fCtxMenu = new TContextMenu("", "");

   Layout();
   MapSubwindows();
   MapWindow();
}

//______________________________________________________________________________
FWListWidget::~FWListWidget()
{
   // Destructor.

   DisconnectSignals();

   delete fCtxMenu;

   // Should un-register editor, all items and list-tree from gEve ... eventually.

   delete fListTree;
   //delete fLTCanvas;
   //delete fLTFrame;
   //delete fFrame;
}

//______________________________________________________________________________
void FWListWidget::ConnectSignals()
{
   // Connect list-tree signals.

   fListTree->Connect("MouseOver(TGListTreeItem*, UInt_t)", "FWListWidget",
                      this, "ItemBelowMouse(TGListTreeItem*, UInt_t)");
   fListTree->Connect("Clicked(TGListTreeItem*, Int_t, UInt_t, Int_t, Int_t)", "FWListWidget",
                      this, "ItemClicked(TGListTreeItem*, Int_t, UInt_t, Int_t, Int_t)");
   fListTree->Connect("DoubleClicked(TGListTreeItem*, Int_t)", "FWListWidget",
                      this, "ItemDblClicked(TGListTreeItem*, Int_t)");
   fListTree->Connect("KeyPressed(TGListTreeItem*, ULong_t, ULong_t)", "FWListWidget",
                      this, "ItemKeyPress(TGListTreeItem*, UInt_t, UInt_t)");

   fSignalsConnected = kTRUE;
}

//______________________________________________________________________________
void FWListWidget::DisconnectSignals()
{
   // Disconnect list-tree signals.

   if (!fSignalsConnected) return;

   fListTree->Disconnect("MouseOver(TGListTreeItem*, UInt_t)",
                         this, "ItemBelowMouse(TGListTreeItem*, UInt_t)");
   fListTree->Disconnect("Clicked(TGListTreeItem*, Int_t, UInt_t, Int_t, Int_t)",
                         this, "ItemClicked(TGListTreeItem*, Int_t, UInt_t, Int_t, Int_t)");
   fListTree->Disconnect("DoubleClicked(TGListTreeItem*, Int_t)",
                         this, "ItemDblClicked(TGListTreeItem*, Int_t)");
   fListTree->Disconnect("KeyPressed(TGListTreeItem*, ULong_t, ULong_t)",
                         this, "ItemKeyPress(TGListTreeItem*, UInt_t, UInt_t)");

   fSignalsConnected = kFALSE;
}

/******************************************************************************/


//______________________________________________________________________________
void FWListWidget::ItemBelowMouse(TGListTreeItem *entry, UInt_t /*mask*/)
{
   // Different item is below mouse.

   TEveElement* el = entry ? (TEveElement*) entry->GetUserData() : 0;
   gEve->GetHighlight()->UserPickedElement(el, kFALSE);
}

//______________________________________________________________________________
void FWListWidget::ItemClicked(TGListTreeItem *item, Int_t btn, UInt_t mask, Int_t x, Int_t y)
{
   // Item has been clicked, based on mouse button do:
   // M1 - select, show in editor;
   // M2 - paste (call gEve->ElementPaste();
   // M3 - popup context menu.

   //printf("ItemClicked item %s List %d btn=%d, x=%d, y=%d\n",
   //  item->GetText(),fDisplayFrame->GetList()->GetEntries(), btn, x, y);

   static const TEveException eh("FWListWidget::ItemClicked ");

   TEveElement* el = (TEveElement*) item->GetUserData();
   if (el == 0) return;
   TObject* obj = el->GetObject(eh);

   switch (btn)
   {
      case 1 :
         gEve->GetSelection()->UserPickedElement(el, mask & kKeyControlMask);
         break;

      case 2:
         if (gEve->ElementPaste(el))
            gEve->Redraw3D();
         break;

      case 3:
         // If control pressed, show menu for render-element itself.
         // event->fState & kKeyControlMask
         // ??? how do i get current event?
         // !!!!! Have this now ... fix.
         if (obj) fCtxMenu->Popup(x, y, obj);
         break;

      default:
         break;
   }
}

//______________________________________________________________________________
void FWListWidget::ItemDblClicked(TGListTreeItem* item, Int_t btn)
{
   // Item has been double-clicked, potentially expand the children.

   static const TEveException eh("FWListWidget::ItemDblClicked ");

   if (btn != 1) return;

   TEveElement* el = (TEveElement*) item->GetUserData();
   if (el == 0) return;

   el->ExpandIntoListTree(fListTree, item);

}

//______________________________________________________________________________
void FWListWidget::ItemKeyPress(TGListTreeItem *entry, UInt_t keysym, UInt_t mask)
{
   // A key has been pressed for an item.
   //
   // Only <Delete>, <Enter> and <Return> keys are handled here,
   // otherwise the control is passed back to TGListTree.

   static const TEveException eh("FWListWidget::ItemKeyPress ");

   entry = fListTree->GetCurrent();
   if (entry == 0) return;

   TEveElement* el = (TEveElement*) entry->GetUserData();

   fListTree->SetEventHandled(); // Reset back to false in default case.

   switch (keysym)
   {
      /*
         case kKey_Delete:
         {
         if (entry->GetParent())
         {
            throw(eh + "DestroyDenied set for this item.");

         TEveElement* parent = (TEveElement*) entry->GetParent()->GetUserData();

         if (parent)
         {
            gEve->RemoveElement(el, parent);
            gEve->Redraw3D();
         }
         }
         else
         {
            throw(eh + "DestroyDenied set for this top-level item.");
         gEve->RemoveFromListTree(el, fListTree, entry);
         gEve->Redraw3D();
         }
         break;
         }
       */
      case kKey_Enter:
      case kKey_Return:
      {
         gEve->GetSelection()->UserPickedElement(el, mask & kKeyControlMask);
         break;
      }

      default:
      {
         fListTree->SetEventHandled(kFALSE);
         break;
      }
   }
}
