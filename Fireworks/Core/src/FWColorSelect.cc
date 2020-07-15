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
#include <boost/bind.hpp>

#include "TGLayout.h"
#include "TGClient.h"
#include "TGGC.h"
#include "TGColorDialog.h"
#include "TColor.h"
#include "TVirtualX.h"

#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/interface/FWColorManager.h"

//------------------------------Constants------------------------------//

enum FWCSelConstants {
  kCFWidth = 15,
  kCFPadLeft = 4,
  kCFPadAbove = 8,
  kCFHeight = 15,
  kCFPadRight = 4,
  kCFPadBelow = 8,

  kHLOffsetX = 3,
  kHLExtraWidth = 5,
  kHLOffsetY = 3,
  kHLExtraHeight = 5,

  kCROffsetX = 5,
  kCRPadAbove = 3,
  kCROffsetY = 5,
  kCRPadBelow = 6,

  kColorPopupGray = 0xcccccc
};

//------------------------------FWColorFrame------------------------------//

FWColorFrame::FWColorFrame(const TGWindow *p, Color_t ci) : TGFrame(p, 20, 20, kOwnBackground) {
  SetColor(ci);
  Resize(kCFWidth, kCFHeight);
}

Bool_t FWColorFrame::HandleButton(Event_t *event) {
  if (event->fType == kButtonRelease) {
    ColorSelected(fColor);
  }
  return kTRUE;
}

void FWColorFrame::SetColor(Color_t ci) {
  fColor = ci;
  SetBackgroundColor(TColor::Number2Pixel(fColor));
}

void FWColorFrame::ColorSelected(Color_t ci) { Emit("ColorSelected(Color_t)", ci); }

//------------------------------FWColorRow------------------------------//

FWColorRow::FWColorRow(const TGWindow *p)
    : TGHorizontalFrame(p, 10, 10, kOwnBackground, TGFrame::GetBlackPixel()), fBackgroundIsBlack(kTRUE) {
  fSelectedIndex = -1;
}

FWColorRow::~FWColorRow() { Cleanup(); }

void FWColorRow::SetBackgroundToBlack(Bool_t toBlack) {
  fBackgroundIsBlack = toBlack;
  if (fBackgroundIsBlack) {
    SetBackgroundColor(TGFrame::GetBlackPixel());
  } else {
    SetBackgroundColor(TGFrame::GetWhitePixel());
  }
}

void FWColorRow::DoRedraw() {
  TGCompositeFrame::DoRedraw();
  DrawHighlight();
}

void FWColorRow::DrawHighlight() {
  if (fSelectedIndex >= 0) {
    Int_t x = fSelectedIndex * (fCc.at(fSelectedIndex)->GetWidth() + kCFPadLeft + kCFPadRight) + kCFPadLeft;
    Int_t y = kCFPadBelow;
    Int_t w = fCc.at(fSelectedIndex)->GetWidth();
    Int_t h = fCc.at(fSelectedIndex)->GetHeight();
    gVirtualX->DrawRectangle(
        fId, GetShadowGC()(), x - kHLOffsetX, y - kHLOffsetY, w + kHLExtraWidth, h + kHLExtraHeight);
  }
}

void FWColorRow::AddColor(Color_t color) {
  fCc.push_back(new FWColorFrame(this, color));
  AddFrame(fCc.back(),
           new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, kCFPadLeft, kCFPadRight, kCFPadAbove, kCFPadBelow));
  fCc.back()->Connect("ColorSelected(Color_t)", "FWColorRow", this, "ColorChanged(Color_t)");
}

void FWColorRow::ResetColor(Int_t iIndex, Color_t iColor) { fCc[iIndex]->SetColor(iColor); }

Int_t FWColorRow::FindColorIndex(Color_t iColor) const {
  Int_t returnValue = -1;
  Int_t index = 0;
  for (std::vector<FWColorFrame *>::const_iterator it = fCc.begin(), itEnd = fCc.end(); it != itEnd; ++it, ++index) {
    if ((*it)->GetColor() == iColor)
      return index;
  }
  return returnValue;
}

void FWColorRow::ColorChanged(Color_t ci) { Emit("ColorChanged(Color_t)", ci); }

//------------------------------FWColorPopup------------------------------//

Bool_t FWColorPopup::fgFreePalette = kFALSE;

Bool_t FWColorPopup::HasFreePalette() { return fgFreePalette; }
void FWColorPopup::EnableFreePalette() { fgFreePalette = kTRUE; }

FWColorPopup::FWColorPopup(const TGWindow *p, Color_t color)
    : TGVerticalFrame(p, 10, 10, kDoubleBorder | kRaisedFrame | kOwnBackground, kColorPopupGray), fShowWheel(kFALSE) {
  SetWindowAttributes_t wattr;
  wattr.fMask = kWAOverrideRedirect;
  wattr.fOverrideRedirect = kTRUE;
  gVirtualX->ChangeWindowAttributes(fId, &wattr);
  AddInput(kStructureNotifyMask);  // to notify the client for structure (i.e. unmap) changes

  fSelectedColor = color;
  fLabel = nullptr;
  fNumColors = 0;

  fFirstRow = new FWColorRow(this);
  fSecondRow = new FWColorRow(this);
  fFirstRow->Connect("ColorChanged(Color_t)", "FWColorPopup", this, "ColorSelected(Color_t)");
  fSecondRow->Connect("ColorChanged(Color_t)", "FWColorPopup", this, "ColorSelected(Color_t)");
}

FWColorPopup::~FWColorPopup() { Cleanup(); }

Bool_t FWColorPopup::HandleButton(Event_t *event) {
  if (event->fX < 0 || event->fX >= (Int_t)fWidth || event->fY < 0 || event->fY >= (Int_t)fHeight) {
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

void FWColorPopup::InitContent(const char *name, const std::vector<Color_t> &colors, bool backgroundIsBlack) {
  fLabel = new TGLabel(this, name);
  fLabel->SetBackgroundColor(GetBackground());
  SetColors(colors, backgroundIsBlack);

  AddFrame(
      fLabel,
      new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, kCROffsetX, kCROffsetY, kCRPadAbove + 1, kCRPadBelow - 1));
  AddFrame(fFirstRow,
           new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, kCROffsetX, kCROffsetY, kCRPadAbove, kCRPadBelow));
  AddFrame(fSecondRow,
           new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, kCROffsetX, kCROffsetY, kCRPadAbove, kCRPadBelow));

  fFirstRow->MapSubwindows();
  fFirstRow->Resize(fFirstRow->GetDefaultSize());

  fSecondRow->MapSubwindows();
  fSecondRow->Resize(fSecondRow->GetDefaultSize());

  if (fgFreePalette) {
    TGTextButton *b = new TGTextButton(this, "Color wheel");
    AddFrame(b,
             new TGLayoutHints(
                 kLHintsTop | kLHintsCenterX | kLHintsExpandX, kCROffsetX, kCROffsetY, kCRPadAbove, 2 * kCRPadBelow));
    b->Connect("Clicked()", "FWColorPopup", this, "PopupColorWheel()");
  }
}

// ----

void FWColorPopup::PopupColorWheel() {
  fShowWheel = kTRUE;
  UnmapWindow();
}

void FWColorPopup::ColorWheelSelected(Pixel_t pix) { ColorSelected(TColor::GetColor(pix)); }

// ----

void FWColorPopup::SetColors(const std::vector<Color_t> &colors, bool backgroundIsBlack) {
  fNumColors = colors.size();
  for (UInt_t i = 0; i < colors.size() / 2; i++) {
    fFirstRow->AddColor(colors.at(i));
  }
  for (UInt_t i = colors.size() / 2; i < colors.size(); i++) {
    fSecondRow->AddColor(colors.at(i));
  }
  fFirstRow->SetBackgroundToBlack(backgroundIsBlack);
  fSecondRow->SetBackgroundToBlack(backgroundIsBlack);
}

void FWColorPopup::ResetColors(const std::vector<Color_t> &colors, bool backgroundIsBlack) {
  fNumColors = colors.size();
  for (UInt_t i = 0; i < colors.size() / 2; i++) {
    fFirstRow->ResetColor(i, colors.at(i));
  }
  fFirstRow->SetBackgroundToBlack(backgroundIsBlack);
  UInt_t index = 0;
  for (UInt_t i = colors.size() / 2; i < colors.size(); i++, ++index) {
    fSecondRow->ResetColor(index, colors.at(i));
  }
  fSecondRow->SetBackgroundToBlack(backgroundIsBlack);
}

void FWColorPopup::PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h) {
  // Popup TGColorPopup at x,y position

  Int_t rx, ry;
  UInt_t rw, rh;

  // Parent is root window for the popup:
  gVirtualX->GetWindowSize(fParent->GetId(), rx, ry, rw, rh);

  if (x < 0)
    x = 0;
  if (x + fWidth > rw)
    x = rw - fWidth;
  if (y < 0)
    y = 0;
  if (y + fHeight > rh)
    y = rh - fHeight;

  MoveResize(x, y, w, h);
  MapSubwindows();
  Layout();
  MapRaised();

  // find out if this is necessary
  gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask | kPointerMotionMask, kNone, kNone);

  gClient->WaitForUnmap(this);
  gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);

  if (fShowWheel) {
    Int_t retc;
    Pixel_t pixel = TColor::Number2Pixel(fSelectedColor);

    TGColorDialog *cd = new TGColorDialog(gClient->GetDefaultRoot(), this, &retc, &pixel, kFALSE);

    cd->Connect("ColorSelected(Pixel_t)", "FWColorPopup", this, "ColorWheelSelected(Pixel_t");

    cd->MapWindow();
    fClient->WaitForUnmap(cd);
    cd->DeleteWindow();

    fShowWheel = kFALSE;
  }
}

void FWColorPopup::SetName(const char *iName) { fLabel->SetText(iName); }

void FWColorPopup::SetSelection(Color_t iColor) {
  fFirstRow->SetSelectedIndex(fFirstRow->FindColorIndex(iColor));
  fSecondRow->SetSelectedIndex(fSecondRow->FindColorIndex(iColor));
}

void FWColorPopup::ColorSelected(Color_t ci) {
  UnmapWindow();
  fSelectedColor = ci;
  Emit("ColorSelected(Color_t)", ci);
}

//------------------------------FWColorSelect------------------------------//

FWColorSelect::FWColorSelect(
    const TGWindow *p, const char *label, Color_t index, const FWColorManager *iManager, Int_t id)
    : TGColorSelect(p, TColor::Number2Pixel(index), id),
      fLabel(label),
      fSelectedColor(index),
      fFireworksPopup(nullptr),
      fColorManager(iManager) {
  std::vector<Color_t> colors;
  fColorManager->fillLimitedColors(colors);

  fFireworksPopup = new FWColorPopup(gClient->GetDefaultRoot(), fColor);
  fFireworksPopup->InitContent(fLabel.c_str(), colors);
  fFireworksPopup->Connect("ColorSelected(Color_t)", "FWColorSelect", this, "SetColorByIndex(Color_t)");

  fColorManager->colorsHaveChanged_.connect(boost::bind(&FWColorSelect::UpdateColors, this));
}

FWColorSelect::~FWColorSelect() { delete fFireworksPopup; }

Bool_t FWColorSelect::HandleButton(Event_t *event) {
  TGFrame::HandleButton(event);
  if (!IsEnabled())
    return kTRUE;

  if (event->fCode != kButton1)
    return kFALSE;

  if ((event->fType == kButtonPress) && HasFocus())
    WantFocus();

  if (event->fType == kButtonPress) {
    fPressPos.fX = fX;
    fPressPos.fY = fY;

    if (fState != kButtonDown) {
      fPrevState = fState;
      SetState(kButtonDown);
    }
  } else {
    if (fState != kButtonUp) {
      SetState(kButtonUp);

      // case when it was dragged during guibuilding
      if ((fPressPos.fX != fX) || (fPressPos.fY != fY)) {
        return kFALSE;
      }

      Window_t wdummy;
      Int_t ax, ay;

      std::vector<Color_t> colors;
      fColorManager->fillLimitedColors(colors);

      fFireworksPopup->ResetColors(colors, fColorManager->backgroundColorIndex() == FWColorManager::kBlackIndex);
      fFireworksPopup->SetSelection(fSelectedColor);

      gVirtualX->TranslateCoordinates(fId, gClient->GetDefaultRoot()->GetId(), 0, fHeight, ax, ay, wdummy);
      fFireworksPopup->PlacePopup(ax, ay, fFireworksPopup->GetDefaultWidth(), fFireworksPopup->GetDefaultHeight());
    }
  }
  return kTRUE;
}

void FWColorSelect::SetColorByIndex(Color_t iColor) { SetColorByIndex(iColor, kTRUE); }

void FWColorSelect::SetColorByIndex(Color_t iColor, Bool_t iSendSignal) {
  fSelectedColor = iColor;
  SetColor(TColor::Number2Pixel(iColor), kFALSE);
  if (iSendSignal) {
    ColorChosen(fSelectedColor);
  }
}

void FWColorSelect::UpdateColors() { SetColor(fSelectedColor, kFALSE); }

void FWColorSelect::ColorChosen(Color_t iColor) { Emit("ColorChosen(Color_t)", iColor); }
