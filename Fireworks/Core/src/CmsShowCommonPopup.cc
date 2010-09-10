#include "TGFont.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGSlider.h"

#include "Fireworks/Core/interface/CmsShowCommonPopup.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"


CmsShowCommonPopup::CmsShowCommonPopup(CmsShowCommon* model, const TGWindow* p, UInt_t w, UInt_t h) :
   TGTransientFrame(gClient->GetDefaultRoot(),p,w,h),
   m_common(model),
   m_backgroundButton(0),
   m_gammaSlider(0),
   m_gammaButton(0)
{
   SetCleanup(kDeepCleanup);

   TGVerticalFrame* vf = new TGVerticalFrame(this);
   AddFrame(vf, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY, 2, 2, 2, 4));

   FWDialogBuilder builder(vf);
   builder.indent(4)
      .expand(false)
      .addLabel("Colors", 14)
      .spaceDown(4)
      .addLabel("Background:", 10)
      .addTextButton("Change Background", &m_backgroundButton, true)
      .spaceDown(4)     
      .addLabel("Brightness:", 10)
      .addHSlider(150, &m_gammaSlider, true)
      .addTextButton("ResetBrightness", &m_gammaButton, true)
      .spaceDown(4)
      .addHSeparator(0)
      .addLabel("Scales ... TODO", 14)
      .spaceDown(20)
      .addHSeparator(0)
      .addLabel("Detector Colors ... (tracker, muon ...) ", 14)
      .spaceDown(20);

   m_backgroundButton->Connect("Clicked()", "CmsShowCommonPopup", this, "switchBackground()");
   m_gammaSlider->Connect("PositionChanged(Int_t)", "CmsShowCommonPopup", this, "setGamma(Int_t)");
   m_gammaSlider->SetRange(-15, 15);
   m_gammaSlider->SetPosition(m_common->gamma());
   m_gammaButton->Connect("Clicked()", "CmsShowCommonPopup", this, "resetGamma()");

   SetWindowName("Common Preferences ...");
   MapSubwindows();
   Resize(GetDefaultSize());
   Layout();
}

CmsShowCommonPopup::~CmsShowCommonPopup()
{
}

void
CmsShowCommonPopup::switchBackground()
{
     m_common->switchBackground();
}
 
void
CmsShowCommonPopup::resetGamma()
{
   m_gammaSlider->SetPosition(0);
   m_common->setGamma(0);
}

void
CmsShowCommonPopup::setGamma(Int_t x)
{
   m_common->setGamma(x);
}
