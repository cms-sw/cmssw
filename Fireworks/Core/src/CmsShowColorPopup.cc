#include "TColor.h"
#include "TG3DLine.h"
#include "TGFont.h"
#include "TClass.h"
#include "TGFrame.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGSlider.h"

// user include files
#include "Fireworks/Core/interface/CmsShowColorPopup.h"
#include "Fireworks/Core/interface/FWColorManager.h"


CmsShowBrightnessPopup::CmsShowBrightnessPopup(const TGWindow* p, UInt_t w, UInt_t h) :
   TGTransientFrame(gClient->GetDefaultRoot(),p,w,h),
   m_colorManager(0),
   m_slider(0)
{
   SetWindowName("Brightness Controller");

   TGVerticalFrame* fr = new TGVerticalFrame(this);
   TGLabel* lb = new TGLabel(fr, "Brightness:");
   lb->SetTextJustify(kTextLeft);
   TGFont* defaultFont = gClient->GetFontPool()->GetFont(lb->GetDefaultFontStruct());
   lb->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
   fr->AddFrame(lb, new TGLayoutHints(kLHintsExpandX));

   m_slider = new TGHSlider(fr, 150);
   m_slider->SetPosition(0);
   m_slider->Connect("PositionChanged(Int_t)", "CmsShowBrightnessPopup", this, "setBrightness(Int_t)");
   m_slider->SetRange(-15, 15);
   fr->AddFrame(m_slider,  new TGLayoutHints(kLHintsNormal, 0, 0, 0, 3));

   TGTextButton* defaultButton = new TGTextButton(fr," Set Default Brightness ");
   defaultButton->Connect("Clicked()","CmsShowBrightnessPopup", this, "defaultBrightness()");
   fr->AddFrame(defaultButton);

   AddFrame(fr, new  TGLayoutHints(kLHintsExpandX, 4,2,0,2));

   MapSubwindows();
   Layout();
}

CmsShowBrightnessPopup::~CmsShowBrightnessPopup()
{
}

void
CmsShowBrightnessPopup::defaultBrightness()
{
   m_colorManager->defaultBrightness();
   m_slider->SetPosition(0);
}

void
CmsShowBrightnessPopup::setBrightness(Int_t x)
{
   m_colorManager->setBrightness(x);
}

void
CmsShowBrightnessPopup::setModel(FWColorManager* mng)
{
   m_colorManager = mng;
   m_slider->SetPosition(m_colorManager->brightness());
}
