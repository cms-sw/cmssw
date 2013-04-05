#ifndef Fireworks_Core_FWColorSelect_h
#define Fireworks_Core_FWColorSelect_h

#include "TGLabel.h"
#include "TGButton.h"
#include "TGColorSelect.h"

#include <vector>

class FWColorManager;
class TGColorPopup;


//------------------------------FWColorFrame------------------------------//

class FWColorFrame : public TGFrame
{
protected:
   Color_t         fColor;

public:
   FWColorFrame(const TGWindow *p=0, Color_t ci=0);
   virtual ~FWColorFrame() {}

   virtual Bool_t HandleButton(Event_t *event);
   virtual void   DrawBorder() {}

   void    SetColor(Color_t);
   Color_t GetColor() const { return fColor; }

   void ColorSelected(Color_t); // *SIGNAL*

   ClassDef(FWColorFrame, 0);

};


//------------------------------FWColorRow------------------------------//

class FWColorRow : public TGHorizontalFrame
{
   void DrawHighlight();

protected:
   Bool_t    fBackgroundIsBlack;
   Int_t     fSelectedIndex;
   std::vector<FWColorFrame *>  fCc;

   virtual void DoRedraw();

public:
   FWColorRow(const TGWindow *p=0);
   virtual ~FWColorRow();

   virtual void AddColor(Color_t color);

   void ResetColor(Int_t, Color_t);
   void SetBackgroundToBlack(Bool_t);
   
   //if it can't find the color it returns -1
   Int_t FindColorIndex(Color_t) const;

   Int_t GetSelectedIndex() const  { return fSelectedIndex; }
   void  SetSelectedIndex(Int_t i) { fSelectedIndex = i; }

   void ColorChanged(Color_t); // *SIGNAL*

   ClassDef(FWColorRow, 0);

};


//------------------------------FWColorPopup------------------------------//

class FWColorPopup : public TGVerticalFrame
{
private:
   void SetColors(const std::vector<Pixel_t>& colors, bool backgroundIsBlack);

protected:
   Color_t     fSelectedColor;
   FWColorRow *fFirstRow, *fSecondRow;
   TGLabel    *fLabel;
   Int_t       fNumColors;
   Bool_t      fShowWheel;

   static Bool_t fgFreePalette;

public:
   FWColorPopup(const TGWindow *p=0, Color_t color=0);
   virtual ~FWColorPopup();

   virtual Bool_t HandleButton(Event_t *event);

   void InitContent(const char *name, const std::vector<Color_t>& colors, bool backgroundIsBlack=true);
   void SetName(const char* iName);
   void SetColors(const std::vector<Color_t>& colors, bool backgroundIsBlack=true);
   void ResetColors(const std::vector<Color_t>& colors, bool backgroundIsBlack=true);
   void SetSelection(Color_t);
   void PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h);

   void ColorSelected(Color_t);          // *SIGNAL*

   void PopupColorWheel();
   void ColorWheelSelected(Pixel_t);

   static Bool_t HasFreePalette();
   static void   EnableFreePalette();

   ClassDef(FWColorPopup, 0);
};


//------------------------------FWColorSelect------------------------------//

class FWColorSelect : public TGColorSelect
{
private:
   std::string           fLabel;
   Color_t               fSelectedColor;
   FWColorPopup         *fFireworksPopup;
   const FWColorManager *fColorManager;
   

public:
   FWColorSelect(const TGWindow *p, const char *label, Color_t colorIndex,
                 const FWColorManager*, Int_t id);
   ~FWColorSelect();

   virtual Bool_t HandleButton(Event_t *event);

   void SetColorByIndex(Color_t iColor);
   void SetColorByIndex(Color_t iColor, Bool_t iSendSignal);
   void UpdateColors();
   const std::string& label() const { return fLabel; } 
   void ColorChosen(Color_t); // *SIGNAL*

   ClassDef(FWColorSelect, 0);

};
#endif
