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
   CmsShowColorPopup( const TGWindow* p = 0, UInt_t w = 1, UInt_t h = 1);
   virtual ~CmsShowColorPopup();

   // ---------- member functions ---------------------------
  void changeBackgroundColor() ;
  void defaultBrightness();
  void setBrightness(int);
   
  void setModel( FWColorManager* mng);

private:
   CmsShowColorPopup(const CmsShowColorPopup&);
   const CmsShowColorPopup& operator=(const CmsShowColorPopup&);

   // ---------- member data --------------------------------

   FWColorManager* m_colorManager;

   TGTextButton*   m_bgButton;
   TGSlider*       m_slider;
};


#endif
