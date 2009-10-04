#ifndef Fireworks_Core_CmsShowColorPopup_h
#define Fireworks_Core_CmsShowColorPopup_h

// system include files
#include "GuiTypes.h"
#include "TGFrame.h"

class TGSlider;
class TGLabel;
class TGTextButton;
class FWColorManager;

class CmsShowBrightnessPopup : public TGTransientFrame
{
public:
   CmsShowBrightnessPopup( const TGWindow* p = 0, UInt_t w = 1, UInt_t h = 1);
   virtual ~CmsShowBrightnessPopup();

   // ---------- member functions ---------------------------

  virtual void CloseWindow() { UnmapWindow(); }

  void defaultBrightness();
  void setBrightness(int);
   
  void setModel( FWColorManager* mng);

private:
   CmsShowBrightnessPopup(const CmsShowBrightnessPopup&);
   const CmsShowBrightnessPopup& operator=(const CmsShowBrightnessPopup&);

   // ---------- member data --------------------------------

   FWColorManager* m_colorManager;

   TGSlider*       m_slider;
};


#endif
