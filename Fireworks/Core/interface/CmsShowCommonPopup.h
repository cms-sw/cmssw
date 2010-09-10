#ifndef Fireworks_Core_CmsShowCommonPopup_h
#define Fireworks_Core_CmsShowCommonPopup_h

#include "GuiTypes.h"
#include "TGFrame.h"

class TGSlider;
class TGLabel;
class TGTextButton;
class TGCheckButton;
class CmsShowCommon;
class FWColorManager;

class CmsShowCommonPopup : public TGTransientFrame
{
public:
   CmsShowCommonPopup( CmsShowCommon*, const TGWindow* p = 0, UInt_t w = 1, UInt_t h = 1);
   virtual ~CmsShowCommonPopup();

   // ---------- member functions ---------------------------

  virtual void CloseWindow() { UnmapWindow(); }

  void switchBackground();
  void setGamma(int);
  void resetGamma();
   
private:
   CmsShowCommonPopup(const CmsShowCommonPopup&);
   const CmsShowCommonPopup& operator=(const CmsShowCommonPopup&);

   // ---------- member data --------------------------------

   CmsShowCommon  *m_common;

   TGTextButton   *m_backgroundButton;
   TGHSlider      *m_gammaSlider;
   TGTextButton   *m_gammaButton;
};


#endif
