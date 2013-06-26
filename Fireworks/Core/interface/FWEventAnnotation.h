#ifndef Fireworks_Core_FWEventAnnotation_h
#define Fireworks_Core_FWEventAnnotation_h

#include "TGLAnnotation.h"
class FWConfiguration;


namespace fwlite {
   class Event;
}

class FWEventAnnotation : public TGLAnnotation
{
public:
   FWEventAnnotation(TGLViewerBase *view);
   virtual ~FWEventAnnotation();

   virtual void   Render(TGLRnrCtx& rnrCtx);

   //configuration management interface
   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);
  
   void setLevel(long x);
   void setEvent();
  
private:
   FWEventAnnotation(const FWEventAnnotation&); // stop default
   const FWEventAnnotation& operator=(const FWEventAnnotation&); // stop default

   void updateOverlayText();

   int  m_level;
};

#endif
