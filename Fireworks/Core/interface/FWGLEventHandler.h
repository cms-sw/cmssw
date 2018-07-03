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
   FWGLEventHandler(TGWindow *w, TObject *obj, TEveCaloLego* l=nullptr);
   ~FWGLEventHandler() override {}

   void   PopupContextMenu(TGLPhysicalShape* pshp, Event_t *event, Int_t gx, Int_t gy) override;

   Bool_t HandleKey(Event_t *event) override;

   Bool_t HandleButton(Event_t *event) override;

   Bool_t HandleFocusChange(Event_t *event) override;
   Bool_t HandleCrossing(Event_t *event) override;

   sigc::signal<void,Int_t,Int_t> openSelectedModelContextMenu_;

   void setViewer(FWEveView* ev) { m_viewer = ev; }

private:
   FWGLEventHandler(const FWGLEventHandler&) = delete; // stop default
   const FWGLEventHandler& operator=(const FWGLEventHandler&) = delete; // stop default

   FWEveView *m_viewer;
};

#endif
