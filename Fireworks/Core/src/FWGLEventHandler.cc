#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWGUIManager.h"

#include "KeySymbols.h"
#include "TGLViewer.h"
#include "TGLPhysicalShape.h"


FWGLEventHandler::FWGLEventHandler(TGWindow *w, TObject *obj, TEveCaloLego* l):
   TEveLegoEventHandler(w, obj, l)
{
}

void
FWGLEventHandler::PopupContextMenu(TGLPhysicalShape* pshp,  Event_t *event, Int_t gx, Int_t gy)
{
   // Popup context menu.

  
   if (event->fState & kKeyShiftMask && event->fState & kKeyControlMask)
   {
      TGLEventHandler::PopupContextMenu(pshp, event, gx, gy);
      return;
   }

   if (pshp)
   {
      SelectForClicked(event);
      openSelectedModelContextMenu_(gx,gy);
   }
}

