#ifndef Fireworks_Core_FWEventAnnotation_h
#define Fireworks_Core_FWEventAnnotation_h

#include "TGLAnnotation.h"
//#include "TGFontManager.h"

#include "Fireworks/Core/interface/FWLongParameter.h"

namespace fwlite {
   class Event;
}

class FWEventAnnotation :public TGLAnnotation
{
public:
   FWEventAnnotation(TGLViewerBase *view, FWParameterizable* confParent);
   virtual ~FWEventAnnotation();

   virtual void   Render(TGLRnrCtx& rnrCtx);

   void setLevel(long x);
   void setEvent();
  
private:
   FWEventAnnotation(const FWEventAnnotation&); // stop default
   const FWEventAnnotation& operator=(const FWEventAnnotation&); // stop default

   void updateOverlayText();

   const fwlite::Event* m_event;
   int  m_level;
};

#endif
