#include <vector>
#include <TQObject.h>
#include <RQ_OBJECT.h>
#include "TGLabel.h"
#include "TGButton.h"
#include "TGColorSelect.h"

enum FWCSelConstants
{
   kCFWidth  = 15, kCFPadLeft =  4, kCFPadAbove = 8,
   kCFHeight = 15, kCFPadRight = 4, kCFPadBelow = 8,

   kHLOffsetX = 3,  kHLExtraWidth  = 5,
   kHLOffsetY = 3,  kHLExtraHeight = 5,

   kFirstRow = 1, kSecondRow = 2,

   kCROffsetX = 5,   kCRPadAbove = 3,
   kCROffsetY = 5,   kCRPadBelow = 6,

   kColorPopupGray = 0xcccccc
};

//------------------------------FWColorFrame------------------------------//
class FWColorFrame : public TGColorFrame {

   RQ_OBJECT("FWColorFrame")

protected:
   Int_t fIndex;

public:
   FWColorFrame(const TGWindow *p = 0, Pixel_t c = 0, Int_t i = 0);
   virtual ~FWColorFrame() {
   }

   virtual Bool_t  HandleButton(Event_t *event);
   virtual void DrawBorder() {
   };

   Int_t GetIndex() const {
      return fIndex;
   }
   void ColorSelected(Int_t frameindex); // *SIGNAL*

   ClassDef(FWColorFrame, 0);

};

//------------------------------FWColorRow------------------------------//
class FWColorRow : public TGCompositeFrame {

   RQ_OBJECT("FWColorRow")

private:
   Int_t fRowIndex;

   void DrawHighlight();

protected:
   Bool_t fIsActive;
   Int_t fSelectedIndex;
   Pixel_t fSelectedColor;
   std::vector<FWColorFrame *>  fCc;

   virtual void DoRedraw();

public:
   FWColorRow(const TGWindow *p = 0, Int_t rowindex = 0);
   virtual ~FWColorRow();

   virtual void RowActive(Bool_t onoff);
   virtual void SetActive(Int_t newat);
   virtual void AddColor(Pixel_t color);

   //if it can't find the color it returns -1
   Int_t FindColorIndex(Pixel_t) const;
   Int_t GetRowIndex() {
      return fRowIndex;
   }
   Int_t GetSelectedIndex() {
      return fSelectedIndex;
   }
   Pixel_t GetSelectedColor() {
      return fSelectedColor;
   }
   void ColorChanged(Int_t newcolor); // *SIGNAL*

   ClassDef(FWColorRow, 0);

};

//------------------------------FWColorPopup------------------------------//
class FWColorPopup : public TGCompositeFrame {

   RQ_OBJECT("FWColorPopup")

private:
   void SetColors(std::vector<Pixel_t> colors);

protected:
   Pixel_t fSelectedColor;
   FWColorRow *fFirstRow, *fSecondRow, *fSelectedRow;
   Int_t fSelectedIndex;
   TGLabel *fLabel;

public:
   FWColorPopup(const TGWindow *p = 0, Pixel_t color = 0);
   virtual ~FWColorPopup();

   virtual Bool_t HandleButton(Event_t *event);

   void InitContent(TGString *name, std::vector<Pixel_t> colors);
   void SetName(const char* iName);
   void SetSelection(Pixel_t);
   void PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void EndPopup();
   FWColorRow *GetActiveRow() {
      return fSelectedRow;
   }
   void ColorBookkeeping(Int_t row); // *SIGNAL*

   ClassDef(FWColorPopup, 0);

};

//------------------------------FWColorSelect------------------------------//
class FWColorSelect : public TGColorSelect {

protected:
   TGString *fLabel;
   std::vector<Pixel_t> fPalette;
   FWColorPopup *fFireworksPopup;

public:
   FWColorSelect(const TGWindow *p, TGString *label, ULong_t color, std::vector<ULong_t> palette, Int_t id);
   ~FWColorSelect();

   virtual Bool_t HandleButton(Event_t *event);

   void CatchSignal(Pixel_t newcolor);

   ClassDef(FWColorSelect, 0);

};
