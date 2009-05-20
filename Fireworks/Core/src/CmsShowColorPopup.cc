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


CmsShowColorPopup::CmsShowColorPopup(const TGWindow* p, UInt_t w, UInt_t h) :
   TGTransientFrame(gClient->GetDefaultRoot(),p,w,h),
   m_colorManager(0),
   m_bgButton(0),
   m_slider(0)
{
   SetWindowName("Color Controller");

   {
      TGVerticalFrame* fr = new TGVerticalFrame(this);
      TGLabel* bgLabel = new TGLabel(fr, "BackgroundColor: ");
      TGFont* defaultFont = gClient->GetFontPool()->GetFont(bgLabel->GetDefaultFontStruct());
      bgLabel->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
      bgLabel->SetTextJustify(kTextLeft);
      fr->AddFrame(bgLabel, new TGLayoutHints(kLHintsExpandX, 0, 2,4));

      m_bgButton = new TGTextButton(fr, " Set White Background ");
      m_bgButton->Connect("Clicked()","CmsShowColorPopup", this, "changeBackgroundColor()");
      fr->AddFrame(m_bgButton);

      AddFrame(fr, new  TGLayoutHints(kLHintsExpandX, 4 ,2,2,2));
   }

   AddFrame(new TGHorizontal3DLine(this, 200, 5), new TGLayoutHints(kLHintsNormal, 4, 4, 4, 4));

   {
      TGVerticalFrame* fr = new TGVerticalFrame(this);
      TGLabel* lb = new TGLabel(fr, "Brightness:");
      lb->SetTextJustify(kTextLeft);
      TGFont* defaultFont = gClient->GetFontPool()->GetFont(lb->GetDefaultFontStruct());
      lb->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
      fr->AddFrame(lb, new TGLayoutHints(kLHintsExpandX));

      m_slider = new TGHSlider(fr, 150);
      m_slider->SetPosition(0);
      m_slider->Connect("PositionChanged(Int_t)", "CmsShowColorPopup", this, "setBrightness(Int_t)");
      m_slider->SetRange(-15, 15);
      fr->AddFrame(m_slider,  new TGLayoutHints(kLHintsNormal, 0, 0, 0, 3));

      TGTextButton* defaultButton = new TGTextButton(fr," Set Default Brightness ");
      defaultButton->Connect("Clicked()","CmsShowColorPopup", this, "defaultBrightness()");
      fr->AddFrame(defaultButton);

      AddFrame(fr, new  TGLayoutHints(kLHintsExpandX, 4,2,0,2));
   }

   MapSubwindows();
   Layout();
}

CmsShowColorPopup::~CmsShowColorPopup()
{
}

//______________________________________________________________________________
void
CmsShowColorPopup::changeBackgroundColor() 
{
   if(FWColorManager::kBlackIndex == m_colorManager->backgroundColorIndex()) {
      m_colorManager->setBackgroundColorIndex(FWColorManager::kWhiteIndex);
      m_bgButton->SetText("Set Black Boackground");
   } else {
      m_colorManager->setBackgroundColorIndex(FWColorManager::kBlackIndex);
      m_bgButton->SetText("Set White Background");
   }
}

void
CmsShowColorPopup::defaultBrightness()
{
   m_colorManager->defaultBrightness();
   m_slider->SetPosition(0);
}

void
CmsShowColorPopup::setBrightness(Int_t x)
{
   m_colorManager->setBrightness(x);
}

void
CmsShowColorPopup::setModel(FWColorManager* mng)
{
   m_colorManager = mng;
   bool bgBlack =   (FWColorManager::kBlackIndex == m_colorManager->backgroundColorIndex());
   m_bgButton->SetText(bgBlack ? " Set White Boackground " : " Set Black Background ");
   m_slider->SetPosition(m_colorManager->brightness());
}
