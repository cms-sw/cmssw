#ifndef Fireworks_Core_FWGLEventHandler_h
#define Fireworks_Core_FWGLEventHandler_h

#include "TEveLegoEventHandler.h"
#include <sigc++/signal.h>

class TGWindow;
class TGLPhysicalShape;
class FWEveView;

class FWGLEventHandler : public TEveLegoEventHandler
{
public:
   FWGLEventHandler(TGWindow *w, TObject *obj, TEveCaloLego* l=0);
   virtual ~FWGLEventHandler() {}

   virtual void   PopupContextMenu(TGLPhysicalShape* pshp, Event_t *event, Int_t gx, Int_t gy);

   virtual Bool_t HandleKey(Event_t *event);

   virtual Bool_t HandleButton(Event_t *event);

   virtual Bool_t HandleFocusChange(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);

   sigc::signal<void,Int_t,Int_t> openSelectedModelContextMenu_;

   void setViewer(FWEveView* ev) { m_viewer = ev; }

private:
   FWGLEventHandler(const FWGLEventHandler&); // stop default
   const FWGLEventHandler& operator=(const FWGLEventHandler&); // stop default

   FWEveView *m_viewer;
};

#endif
