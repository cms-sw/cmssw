#ifndef Fireworks_Core_FWNumberEntryField_h
#define Fireworks_Core_FWNumberEntryField_h

#include "TGNumberEntry.h"

//------------------------------FWNumberEntryField------------------------------//

class FWNumberEntryField : public TGNumberEntryField
{
private:

public:
   FWNumberEntryField(const TGWindow *p, Int_t id,
                      Double_t val, GContext_t norm,
                      FontStruct_t font = GetDefaultFontStruct(),
                      UInt_t option = kSunkenFrame | kDoubleBorder,
                      Pixel_t back = GetWhitePixel());
   FWNumberEntryField(const TGWindow *parent = 0,
                      Int_t id = -1, Double_t val = 0,
                      EStyle style = kNESReal,
                      EAttribute attr = kNEAAnyNumber,
                      ELimit limits = kNELNoLimits,
                      Double_t min = 0, Double_t max = 1);

   virtual ~FWNumberEntryField() {}

   virtual Bool_t HandleFocusChange(Event_t* event);
   virtual void   ReturnPressed();

   virtual UInt_t GetUIntNumber();
   virtual void   SetUIntNumber(UInt_t n);

   ClassDef(FWNumberEntryField, 0);

};
#endif
