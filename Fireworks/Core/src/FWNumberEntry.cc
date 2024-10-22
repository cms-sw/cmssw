#include "Fireworks/Core/src/FWNumberEntry.h"

#include <cstdlib>

//------------------------------FWNumberEntryField------------------------------//

//______________________________________________________________________________
FWNumberEntryField::FWNumberEntryField(
    const TGWindow* p, Int_t id, Double_t val, GContext_t norm, FontStruct_t font, UInt_t option, ULong_t back)
    : TGNumberEntryField(p, id, val, norm, font, option, back) {
  // Constructs a number entry field.
}

//______________________________________________________________________________
FWNumberEntryField::FWNumberEntryField(const TGWindow* parent,
                                       Int_t id,
                                       Double_t val,
                                       EStyle style,
                                       EAttribute attr,
                                       ELimit limits,
                                       Double_t min,
                                       Double_t max)
    : TGNumberEntryField(parent, id, val, style, attr, limits, min, max) {
  // Constructs a number entry field.
}

//------------------------------------------------------------------------------

//______________________________________________________________________________
Bool_t FWNumberEntryField::HandleFocusChange(Event_t* event) {
  // Handle focus change.
  // Avoid verification by TGNumberEntryField (which is f***ed).

  return TGTextEntry::HandleFocusChange(event);
}

//______________________________________________________________________________
void FWNumberEntryField::ReturnPressed() {
  // Return was pressed.
  // Avoid verification by TGNumberEntryField (which is f***ed).

  TGTextEntry::ReturnPressed();
}

//------------------------------------------------------------------------------

//______________________________________________________________________________
UInt_t FWNumberEntryField::GetUIntNumber() { return static_cast<UInt_t>(strtoul(GetText(), nullptr, 10)); }

//______________________________________________________________________________
void FWNumberEntryField::SetUIntNumber(UInt_t n) { SetText(Form("%u", n), kFALSE); }

//______________________________________________________________________________
ULong64_t FWNumberEntryField::GetULong64Number() { return static_cast<ULong64_t>(strtoull(GetText(), nullptr, 10)); }

//______________________________________________________________________________
void FWNumberEntryField::SetULong64Number(ULong64_t n) { SetText(Form("%llu", n), kFALSE); }
