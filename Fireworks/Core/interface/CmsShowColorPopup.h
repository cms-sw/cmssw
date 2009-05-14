#ifndef Fireworks_Core_CmsShowColorPopup_h
#define Fireworks_Core_CmsShowColorPopup_h

// system include files
#include "GuiTypes.h"
#include "TGFrame.h"

class TGSlider;
class TGLabel;
class TGTextButton;
class FWColorManager;

class CmsShowColorPopup : public TGTransientFrame
{
public:
   CmsShowColorPopup( FWColorManager*, const TGWindow* p = 0, UInt_t w = 1, UInt_t h = 1);
   virtual ~CmsShowColorPopup();

   // ---------- member functions ---------------------------
  void changeBackgroundColor() ;
  void defaultBrightness();
  void setBrightness(Int_t);

private:
   CmsShowColorPopup(const CmsShowColorPopup&);
   const CmsShowColorPopup& operator=(const CmsShowColorPopup&);

   // ---------- member data --------------------------------
   TGLabel* m_bgLabel;
   TGTextButton* m_bgButton;

   TGTextButton* m_defaultButton;
   TGSlider*         m_slider;

   FWColorManager* m_colorManager;
};


#endif
