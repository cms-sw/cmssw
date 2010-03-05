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

   //   void setEvent( const fwlite::Event* event);
   void setEvent();
   FWLongParameter*  m_level;

   virtual void   Render(TGLRnrCtx& rnrCtx);
   void updateText();
  
private:
   FWEventAnnotation(const FWEventAnnotation&); // stop default
   const FWEventAnnotation& operator=(const FWEventAnnotation&); // stop default

   const fwlite::Event* m_event;
};

#endif
