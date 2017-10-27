#ifndef Fireworks_Core_CmsShowCommonPopup_h
#define Fireworks_Core_CmsShowCommonPopup_h

#ifndef __CINT__
#include <memory>
#endif
#include "GuiTypes.h"
#include "TGFrame.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"
#ifndef __CINT__
#include "Fireworks/Core/interface/FWColorManager.h"
#endif

class TGHSlider;
class TGLabel;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class CmsShowCommon;
class FWColorManager;
class FWColorSelect;
class FWParameterBase;
class FWParameterSetterBase;

class CmsShowCommonPopup : public TGTransientFrame,
                           public FWParameterSetterEditorBase
{
public:
   CmsShowCommonPopup( CmsShowCommon*, const TGWindow* p = nullptr, UInt_t w = 1, UInt_t h = 1);
   ~CmsShowCommonPopup() override;

   // ---------- member functions ---------------------------

   void CloseWindow() override { UnmapWindow(); }

   void switchBackground();
   void permuteColors();
   void randomizeColors();

   void changeGeomColor(Color_t);
   void changeGeomTransparency2D(int);
   void changeGeomTransparency3D(int);
   void changeSelectionColorSet(Color_t);
   void colorSetChanged();
   void setPaletteGUI();

   TGComboBox* getCombo() {return m_combo;}
   ClassDefOverride(CmsShowCommonPopup, 0);
 
private:
   CmsShowCommonPopup(const CmsShowCommonPopup&);
   const CmsShowCommonPopup& operator=(const CmsShowCommonPopup&);

   TGFrame* makeSetter(TGCompositeFrame* frame, FWParameterBase* param);
   void getColorSetColors (int& hci, int& sci);
   // ---------- member data --------------------------------

   CmsShowCommon  *m_common;

   TGTextButton   *m_backgroundButton;
   TGHSlider      *m_gammaSlider;
   TGTextButton   *m_gammaButton;
#ifndef __CINT__
   FWColorSelect* m_colorSelectWidget[kFWGeomColorSize];
   FWColorSelect* m_colorRnrCtxHighlightWidget;   
   FWColorSelect* m_colorRnrCtxSelectWidget;
   std::vector<std::shared_ptr<FWParameterSetterBase> > m_setters;
#endif
   TGComboBox     *m_combo;  
};



#endif
