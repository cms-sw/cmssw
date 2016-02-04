#include "TGFont.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGSlider.h"
#include "TGComboBox.h"
#include "TGClient.h"

#include "Fireworks/Core/interface/CmsShowCommonPopup.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/interface/FWViewEnergyScaleEditor.h"

#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

#include "Fireworks/Core/interface/FWParameterSetterBase.h"

CmsShowCommonPopup::CmsShowCommonPopup(CmsShowCommon* model, const TGWindow* p, UInt_t w, UInt_t h) :
   TGTransientFrame(gClient->GetDefaultRoot(),p,w,h),
   m_common(model),
   m_backgroundButton(0),
   m_gammaSlider(0),
   m_gammaButton(0)
{
   SetCleanup(kDeepCleanup);
   
   //
   // scales
   //
   TGCompositeFrame* vf2 = new TGVerticalFrame(this);
   AddFrame(vf2, new TGLayoutHints(kLHintsNormal, 2, 2, 2, 2));
 
   FWDialogBuilder bS(vf2);
   bS.addLabel("GlobalScales", 14).vSpacer(5);

   FWViewEnergyScaleEditor* scaleEditor = new FWViewEnergyScaleEditor(m_common->m_energyScale.get(), vf2);
   vf2->AddFrame(scaleEditor);

   //   addParamSetter( &m_common->m_energyCombinedSwitch, vf2, "CombinedMode");
   //
   // brigtness
   //
   TGCompositeFrame* vf = new TGVerticalFrame(this);
   AddFrame(vf, new TGLayoutHints(kLHintsNormal, 2, 2, 2, 4));
   FWDialogBuilder builder(vf);
   builder.indent(3)
      .addHSeparator(0)
      .addLabel("General Colors:", 14)
      .spaceDown(4).expand(false)  
      .addTextButton("Black/White Background", &m_backgroundButton).expand(false)
      .spaceDown(4)  
      .addLabel("Brightness:", 10, 0).expand(false).floatLeft()
      .addHSlider(120, &m_gammaSlider).expand(false)
      .addTextButton("Reset Brightness", &m_gammaButton).expand(false)
      .spaceDown(4)
      .addHSeparator(0)
      .addLabel("Detector Colors: ", 14)
      .spaceDown(2)
      .indent(0);

   m_backgroundButton->SetEnabled(true);
   m_gammaButton->SetEnabled(true);
   m_gammaSlider->SetEnabled(true);

   TGFont* smallFont = 0;
   FontStruct_t defaultFontStruct = m_backgroundButton->GetDefaultFontStruct();
   try
   { 
      TGFontPool *pool = gClient->GetFontPool();
      TGFont* defaultFont = pool->GetFont(defaultFontStruct);
      FontAttributes_t attributes = defaultFont->GetFontAttributes();
      smallFont = pool->GetFont(attributes.fFamily, 8,  attributes.fWeight, attributes.fSlant);                                      
   } 
   catch(...)
   {
      // Ignore exceptions.
   }

   //
   // geom colors
   //

   TGHSlider* transpWidget2D = 0;
   TGHSlider* transpWidget3D = 0;
   TGCompositeFrame* top  = new TGVerticalFrame(vf);
   vf->AddFrame(top, new TGLayoutHints(kLHintsNormal, 2, 2, 2, 4));
   {
      TGHorizontalFrame* hf = new TGHorizontalFrame(top); 
      hf->AddFrame(new TGLabel(hf, "Tansparency 2D:"), new TGLayoutHints(kLHintsNormal, 2,  2, 3, 3));
      transpWidget2D = new TGHSlider(hf, 100, kSlider1);
      hf->AddFrame( transpWidget2D);
      top->AddFrame(hf,new TGLayoutHints( kLHintsNormal, 2,  2, 3, 3));
   }
   {
      TGHorizontalFrame* hf = new TGHorizontalFrame(top); 
      hf->AddFrame(new TGLabel(hf, "Tansparency 3D:") , new TGLayoutHints(kLHintsNormal,2, 2, 3, 3));
      transpWidget3D = new TGHSlider(hf, 100, kSlider1);
      hf->AddFrame( transpWidget3D);
      top->AddFrame(hf, new TGLayoutHints(kLHintsExpandX, 0, 0, 0, 10));
   }

   std::string names[kFWGeomColorSize];
   names[kFWPixelBarrelColorIndex   ] = "Pixel Barrel";
   names[kFWPixelEndcapColorIndex   ] = "Pixel Endcap" ;
   names[kFWTrackerBarrelColorIndex ] = "Tracker Barrel";
   names[kFWTrackerEndcapColorIndex ] = "Tracker Endcap";
   names[kFWMuonBarrelLineColorIndex] = "Muon Barrel";
   names[kFWMuonEndcapLineColorIndex] = "Muon Endcap";
   int i = 0;
   for (int k = 0; k < 3; ++k)
   {
      TGHorizontalFrame* hf = new TGHorizontalFrame(top); 
      top->AddFrame(hf);

      for (int j = 0 ; j < 2; ++j)
      {
         m_colorSelectWidget[i] = new FWColorSelect(hf, names[i].c_str(), 0, m_common->colorManager(), i);
         hf->AddFrame(m_colorSelectWidget[i]); 
         m_colorSelectWidget[i]->SetColorByIndex(m_common->colorManager()->geomColor(FWGeomColorIndex(i)) ,kFALSE);
         m_colorSelectWidget[i]->Connect("ColorChosen(Color_t)", "CmsShowCommonPopup", this, "changeGeomColor(Color_t)");

         TGFrame* lf = new TGHorizontalFrame(hf, 100, 16, kFixedSize);
         TGLabel* label = new TGLabel(lf, m_colorSelectWidget[i]->label().c_str());
         label->SetTextFont(smallFont);
         hf->AddFrame(lf); 

         ++i;
      }
   }

   m_backgroundButton->Connect("Clicked()", "CmsShowCommonPopup", this, "switchBackground()");
   m_gammaSlider->Connect("PositionChanged(Int_t)", "CmsShowCommonPopup", this, "setGamma(Int_t)");
   m_gammaSlider->SetRange(-15, 15);
   m_gammaSlider->SetPosition(m_common->gamma());
   m_gammaButton->Connect("Clicked()", "CmsShowCommonPopup", this, "resetGamma()");

   transpWidget2D->SetRange(0, 100);
   transpWidget2D->SetPosition(m_common->colorManager()->geomTransparency(true));
   transpWidget2D->Connect("PositionChanged(Int_t)", "CmsShowCommonPopup", this, "changeGeomTransparency2D(Int_t)");

   transpWidget3D->SetRange(0, 100);
   transpWidget3D->SetPosition(m_common->colorManager()->geomTransparency(false));
   transpWidget3D->Connect("PositionChanged(Int_t)", "CmsShowCommonPopup", this, "changeGeomTransparency3D(Int_t)");

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
CmsShowCommonPopup::changeGeomTransparency2D(int iTransp)
{
   m_common->setGeomTransparency(iTransp, true);
}

void
CmsShowCommonPopup::changeGeomTransparency3D(int iTransp)
{
   m_common->setGeomTransparency(iTransp, false);
}


/* Called by FWGUIManager when change background. */
void 
CmsShowCommonPopup::colorSetChanged()
{
   for (int i = 0 ; i < kFWGeomColorSize; ++i)
      m_colorSelectWidget[i]->SetColorByIndex(m_common->colorManager()->geomColor(FWGeomColorIndex(i)), kFALSE);
   
}
