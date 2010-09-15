#include "TGFont.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGSlider.h"
#include "TGClient.h"

#include "Fireworks/Core/interface/CmsShowCommonPopup.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"
#include "Fireworks/Core/src/FWColorSelect.h"


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
   TGHSlider* m_transpWidget;

   TGLabel* smallLabel;

   FWDialogBuilder builder(vf);
   builder.indent(3)
      .spaceDown(3)  
      .addLabel("General Colors:", 14)
      .spaceDown(4)  
      .addTextButton("Black/White Background", &m_backgroundButton)
      .spaceDown(4)  
      .addLabel("Brightness:", 8, 0, &smallLabel)
      .addHSlider(150, &m_gammaSlider)
      .addTextButton("Reset Brightness", &m_gammaButton)
      .spaceDown(4)
      .addHSeparator(0)
      .addLabel("Detector Colors: ", 14)
      .spaceDown(2)
      .indent(0)
      .spaceDown(1)  
      .addLabel("Transparency:", 8)
      .addHSlider(100, &m_transpWidget)
      .spaceDown(2);
     
   m_backgroundButton->SetEnabled(true);
   m_gammaButton->SetEnabled(true);
   m_transpWidget->SetEnabled(true);

   TGCompositeFrame* tp  = (TGCompositeFrame*)m_gammaButton->GetParent()->GetParent();
   TGHorizontalFrame* parent[3];
   for (int i = 0; i < kFWGeomColorSize; ++i)
   {
      parent[i] = new TGHorizontalFrame(tp);
      tp->AddFrame(parent[i], new TGLayoutHints(kLHintsExpandX| kLHintsTop));
   }

   m_colorSelectWidget[kFWMuonBarrelLineColorIndex] = new FWColorSelect(parent[kFWMuonBarrelLineColorIndex], "Muon Barrel" , 0, m_common->colorManager(), kFWMuonBarrelLineColorIndex);
   m_colorSelectWidget[kFWMuonEndcapLineColorIndex] = new FWColorSelect(parent[kFWMuonEndcapLineColorIndex], "Muon Endcap" , 0, m_common->colorManager(), kFWMuonEndcapLineColorIndex);
   m_colorSelectWidget[kFWTrackerColorIndex       ] = new FWColorSelect(parent[kFWTrackerColorIndex],        "Tracker"     , 0, m_common->colorManager(), kFWTrackerColorIndex);


   for (int i = 0 ; i < kFWGeomColorSize; ++i)
   {

      parent[i]->AddFrame(m_colorSelectWidget[i]); 
      m_colorSelectWidget[i]->SetColorByIndex(m_common->colorManager()->geomColor(FWGeomColorIndex(i)) ,kFALSE);
      m_colorSelectWidget[i]->Connect("ColorChosen(Color_t)", "CmsShowCommonPopup", this, "changeGeomColor(Color_t)");
      TGLabel* label = new TGLabel(parent[i], m_colorSelectWidget[i]->label().c_str());
      label->SetTextFont(smallLabel->GetFont());
      parent[i]->AddFrame(label); 
   }

   m_backgroundButton->Connect("Clicked()", "CmsShowCommonPopup", this, "switchBackground()");
   m_gammaSlider->Connect("PositionChanged(Int_t)", "CmsShowCommonPopup", this, "setGamma(Int_t)");
   m_gammaSlider->SetRange(-15, 15);
   m_gammaSlider->SetPosition(m_common->gamma());
   m_gammaButton->Connect("Clicked()", "CmsShowCommonPopup", this, "resetGamma()");

   m_transpWidget->SetRange(0, 100);
   m_transpWidget->SetPosition(m_common->colorManager()->geomTransparency());
   m_transpWidget->Connect("PositionChanged(Int_t)", "CmsShowCommonPopup", this, "changeGeomTransparency(Int_t)");

   SetWindowName("Common Preferences ...");
   MapSubwindows();
   Resize(GetDefaultSize());
   Layout();
   CenterOnParent(kTRUE, TGTransientFrame::kTopRight);
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

void
CmsShowCommonPopup::changeGeomColor(Color_t iColor)
{
   TGColorSelect *cs = (TGColorSelect *) gTQSender;
   FWGeomColorIndex cidx = FWGeomColorIndex(cs->WidgetId());
   m_common->setGeomColor(cidx, iColor);
}

void
CmsShowCommonPopup::changeGeomTransparency(int iTransp)
{
   m_common->setGeomTransparency(iTransp);
}

/* Called by FWGUIManager when change background. */
void 
CmsShowCommonPopup::colorSetChanged()
{
   for (int i = 0 ; i < kFWGeomColorSize; ++i)
      m_colorSelectWidget[i]->SetColorByIndex(m_common->colorManager()->geomColor(FWGeomColorIndex(i)), kFALSE);
   
}
