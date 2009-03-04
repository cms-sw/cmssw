/*
    All access to this colour picker goes through FWColorSelect.
    Widget creation: new FWColorSelect(p, n, col, cols, id)
                        p: parent window
                        n: the name of the popup
                      col: initially selected colour (in Pixel_t)
                     cols: a vector<Pixel_t> of colour values
                       id: window ID

    After creation, connect to signal method ColorSelected(Pixel_t) in
    FWColorSelect to receive colour changes.
 */

#include "TGColorSelect.h"
#include "TGLayout.h"
#include "TGClient.h"
#include "TGMsgBox.h"
#include "TGGC.h"
#include "TGColorSelect.h"
#include "TGColorDialog.h"
#include "TGResourcePool.h"
#include "TG3DLine.h"
#include "TColor.h"
#include "Riostream.h"

#include "Fireworks/Core/src/FWColorSelect.h"

//------------------------------FWColorFrame------------------------------//
FWColorFrame::FWColorFrame(const TGWindow *p, Pixel_t color, Int_t index) :
   TGColorFrame(p, color)
{
   fIndex = index;
   Resize(kCFWidth, kCFHeight);
}

Bool_t FWColorFrame::HandleButton(Event_t *event)
{
   if (event->fType == kButtonRelease)
   {
      ColorSelected(fIndex);
   }
   return kTRUE;
}

void FWColorFrame::ColorSelected(Int_t frameindex)
{
   Emit("ColorSelected(Int_t)", frameindex);
}

//------------------------------FWColorRow------------------------------//
FWColorRow::FWColorRow(const TGWindow *p, Int_t rowindex) :
   TGCompositeFrame(p, 10, 10, kOwnBackground, TGFrame::GetBlackPixel())
{
   SetLayoutManager(new TGHorizontalLayout(this));

   fRowIndex = rowindex;
   fSelectedIndex = 0;
   fSelectedColor = 0;
   fIsActive = kFALSE;
}

FWColorRow::~FWColorRow()
{
   Cleanup();
}

void FWColorRow::DoRedraw()
{
   TGCompositeFrame::DoRedraw();
   DrawHighlight();
}

void FWColorRow::DrawHighlight()
{
   GContext_t gc;
   if (fIsActive) gc = GetShadowGC() ();
   else gc = GetBlackGC() ();

   Int_t x = fSelectedIndex * (fCc.at(fSelectedIndex)->GetWidth() + kCFPadLeft + kCFPadRight) + kCFPadLeft;
   Int_t y = kCFPadBelow;
   Int_t w = fCc.at(fSelectedIndex)->GetWidth();
   Int_t h = fCc.at(fSelectedIndex)->GetHeight();

   gVirtualX->DrawRectangle(fId, gc, x - kHLOffsetX, y - kHLOffsetY, w + kHLExtraWidth, h + kHLExtraHeight);
}

void FWColorRow::RowActive(Bool_t onoff)
{
   fIsActive = onoff;
}

void FWColorRow::SetActive(Int_t newat)
{
   fCc.at(fSelectedIndex)->SetActive(kFALSE);
   fSelectedIndex = newat;
   fCc.at(fSelectedIndex)->SetActive(kTRUE);
   fSelectedColor = fCc.at(fSelectedIndex)->GetColor();
}

void FWColorRow::AddColor(Pixel_t color)
{
   Int_t pos = fCc.size();
   fCc.push_back(new FWColorFrame(this, color, pos));
   AddFrame(fCc.back(), new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, kCFPadLeft, kCFPadRight, kCFPadAbove, kCFPadBelow));
   fCc.back()->Connect("ColorSelected(Int_t)","FWColorRow", this, "ColorChanged(Int_t)");
}

Int_t 
FWColorRow::FindColorIndex(Pixel_t iColor) const
{
   Int_t returnValue = -1;
   Int_t index=0;
   for(std::vector<FWColorFrame*>::const_iterator it = fCc.begin(), itEnd = fCc.end();
       it!=itEnd;++it,++index) {
      if((*it)->GetColor() == iColor) {
         return index;
      }
   }
   return returnValue;
}

void FWColorRow::ColorChanged(Int_t newcolor)
{
   fIsActive = kTRUE;
   SetActive(newcolor);
   Emit("ColorChanged(Int_t)", fRowIndex);
}

//------------------------------FWColorPopup------------------------------//
FWColorPopup::FWColorPopup(const TGWindow *p, Pixel_t color) :
   TGCompositeFrame(p, 10, 10, kDoubleBorder | kRaisedFrame | kOwnBackground, kColorPopupGray)
{
   SetLayoutManager(new TGVerticalLayout(this));

   SetWindowAttributes_t wattr;
   wattr.fMask = kWAOverrideRedirect;
   wattr.fOverrideRedirect = kTRUE;
   gVirtualX->ChangeWindowAttributes(fId, &wattr);
   AddInput(kStructureNotifyMask); // to notify the client for structure (i.e. unmap) changes

   fSelectedRow = 0;
   fSelectedIndex = -1;
   fSelectedColor = color;
   fLabel = 0;

   fFirstRow = new FWColorRow(this, kFirstRow);
   fSecondRow = new FWColorRow(this, kSecondRow);
   fFirstRow->Connect("ColorChanged(Int_t)", "FWColorPopup", this, "ColorBookkeeping(Int_t)");
   fSecondRow->Connect("ColorChanged(Int_t)", "FWColorPopup", this, "ColorBookkeeping(Int_t)");
}

FWColorPopup::~FWColorPopup()
{
   Cleanup();
}

Bool_t FWColorPopup::HandleButton(Event_t *event)
{
   if (event->fX < 0 || event->fX >= (Int_t) fWidth ||
       event->fY < 0 || event->fY >= (Int_t) fHeight) {
      if (event->fType == kButtonRelease)
         UnmapWindow();
   } else {
      TGFrame *f = GetFrameFromPoint(event->fX, event->fY);
      if (f && f != this) {
         TranslateCoordinates(f, event->fX, event->fY, event->fX, event->fY);
         f->HandleButton(event);
      }
   }
   return kTRUE;
}

void FWColorPopup::InitContent(TGString *name, std::vector<Pixel_t> colors)
{
   fLabel = new TGLabel(this, name);
   fLabel->SetBackgroundColor(GetBackground());
   SetColors(colors);

   AddFrame(fLabel, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, kCROffsetX, kCROffsetY, kCRPadAbove + 1, kCRPadBelow - 1));
   AddFrame(fFirstRow, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, kCROffsetX, kCROffsetY, kCRPadAbove, kCRPadBelow));
   AddFrame(fSecondRow, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, kCROffsetX, kCROffsetY, kCRPadAbove, 2 * kCRPadBelow));

   fFirstRow->MapSubwindows();
   fFirstRow->Resize(fFirstRow->GetDefaultSize());

   fSecondRow->MapSubwindows();
   fSecondRow->Resize(fSecondRow->GetDefaultSize());
}

void FWColorPopup::SetColors(std::vector<Pixel_t> colors)
{
   for (UInt_t i = 0; i < colors.size() / 2; i++)
   {
      fFirstRow->AddColor(colors.at(i));
      if (colors.at(i) == fSelectedColor)
      {
         fSelectedRow = fFirstRow;
         fSelectedIndex = i;
      }
   }
   for (UInt_t i = colors.size() / 2; i < colors.size(); i++)
   {
      fSecondRow->AddColor(colors.at(i));
      if (colors.at(i) == fSelectedColor)
      {
         fSelectedRow = fSecondRow;
         fSelectedIndex = i - colors.size();
      }
   }
   fSelectedRow->RowActive(kTRUE);
   fSelectedRow->SetActive(fSelectedIndex);
}

void FWColorPopup::PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Popup TGColorPopup at x,y position

   Int_t rx, ry;
   UInt_t rw, rh;

   // Parent is root window for the popup:
   gVirtualX->GetWindowSize(fParent->GetId(), rx, ry, rw, rh);

   if (x < 0) x = 0;
   if (x + fWidth > rw) x = rw - fWidth;
   if (y < 0) y = 0;
   if (y + fHeight > rh) y = rh - fHeight;

   MoveResize(x, y, w, h);
   MapSubwindows();
   Layout();
   MapRaised();

   //find out if this is necessary
   gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask | kPointerMotionMask, kNone, kNone);

   gClient->WaitForUnmap(this);
   EndPopup();
}

void FWColorPopup::EndPopup()
{
   // Release pointer
   gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);
}

void FWColorPopup::SetName(const char* iName)
{
   fLabel->SetText(iName);
}

void FWColorPopup::SetSelection(Pixel_t iColor)
{
   FWColorRow* foundIn = fFirstRow;
   Int_t index = fFirstRow->FindColorIndex(iColor);
   if(-1 ==index) {
      foundIn = fSecondRow;
      index = foundIn->FindColorIndex(iColor);
      if(-1==index) {
         std::cout <<"could not find color "<<iColor<<std::endl;
         return;
      }
   }
   if(foundIn != fSelectedRow) {
      fSelectedRow->RowActive(kFALSE);
      fSelectedRow=foundIn;
      fSelectedRow->RowActive(kTRUE);
   }
   fSelectedRow->SetActive(index);
}

void FWColorPopup::ColorBookkeeping(Int_t row)
{
   if (row != fSelectedRow->GetRowIndex())
   {
      fSelectedRow->RowActive(kFALSE);
      if (fSelectedRow == fFirstRow)
      {
         fSelectedRow = fSecondRow;
      }
      else
      {
         fSelectedRow = fFirstRow;
      }
      fSelectedRow->RowActive(kTRUE);
   }
   fSelectedIndex = fSelectedRow->GetSelectedIndex();
   fSelectedColor = fSelectedRow->GetSelectedColor();
   Emit("ColorBookkeeping(Int_t)", fSelectedColor);
   UnmapWindow(); // close the popup...
}

//------------------------------FWColorSelect------------------------------//
FWColorSelect::FWColorSelect(const TGWindow *p, TGString
                             *label, ULong_t color, std::vector<ULong_t> palette, Int_t id) :
   TGColorSelect(p, color, id)
{
   fLabel = label;
   fPalette = palette;
   fFireworksPopup = new FWColorPopup(gClient->GetDefaultRoot(), fColor);
   fFireworksPopup->InitContent(fLabel, fPalette);
   fFireworksPopup->Connect("ColorBookkeeping(Int_t)","FWColorSelect", this, "CatchSignal(Pixel_t)");
}

FWColorSelect::~FWColorSelect()
{
   delete fFireworksPopup;
}

Bool_t FWColorSelect::HandleButton(Event_t *event)
{
   TGFrame::HandleButton(event);
   if (!IsEnabled()) return kTRUE;

   if (event->fCode != kButton1) return kFALSE;

   if ((event->fType == kButtonPress) && HasFocus()) WantFocus();

   if (event->fType == kButtonPress)
   {
      fPressPos.fX = fX;
      fPressPos.fY = fY;

      if (fState != kButtonDown) {
         fPrevState = fState;
         SetState(kButtonDown);
      }
   }
   else
   {
      if (fState != kButtonUp)
      {
         SetState(kButtonUp);

         // case when it was dragged during guibuilding
         if ((fPressPos.fX != fX) || (fPressPos.fY != fY))
         {
            return kFALSE;
         }
         Window_t wdummy;
         Int_t ax, ay;

         gVirtualX->TranslateCoordinates(fId, gClient->GetDefaultRoot()->GetId(), 0, fHeight, ax, ay, wdummy);
         fFireworksPopup->PlacePopup(ax, ay, fFireworksPopup->GetDefaultWidth(), fFireworksPopup->GetDefaultHeight());
      }
   }
   return kTRUE;
}

void FWColorSelect::CatchSignal(Pixel_t newcolor)
{
   SetColor(newcolor, kTRUE);
}
