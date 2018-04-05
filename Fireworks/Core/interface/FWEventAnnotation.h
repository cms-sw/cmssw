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
   ~FWEventAnnotation() override;

   void   Render(TGLRnrCtx& rnrCtx) override;

   //configuration management interface
   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);
  
   void setLevel(long x);
   void setEvent();
  
private:
   FWEventAnnotation(const FWEventAnnotation&) = delete; // stop default
   const FWEventAnnotation& operator=(const FWEventAnnotation&) = delete; // stop default

   void updateOverlayText();

   int  m_level;
};

#endif
