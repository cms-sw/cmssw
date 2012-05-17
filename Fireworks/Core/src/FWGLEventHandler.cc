#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWGUIManager.h"

#include "KeySymbols.h"
#include "TGLViewer.h"
#include "TGLWidget.h"
#include "TGLPhysicalShape.h"
#include "TGLLogicalShape.h"
#include "FWGeoTopNodeScene.h"

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
    
    if (pshp->GetLogical()) 
    {
      FWGeoTopNodeGLScene* js = dynamic_cast<FWGeoTopNodeGLScene*>(pshp->GetLogical()->GetScene());
      if (js) {
        js->GeoPopupMenu(gx, gy);
        return;
      }
    }
    
    openSelectedModelContextMenu_(gx,gy);
  }
}

Bool_t FWGLEventHandler::HandleKey(Event_t *event)
{
   UInt_t keysym;
   char tmp[2];
   gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);

   if (keysym == kKey_Enter || keysym == kKey_Return || keysym == kKey_Space)
   {
      if (event->fType == kGKeyPress)
      {
         Int_t    x, y;
         Window_t childdum;
         gVirtualX->TranslateCoordinates(fGLViewer->GetGLWidget()->GetId(), gClient->GetDefaultRoot()->GetId(),
                                         event->fX, event->fY, x, y, childdum);

         fGLViewer->RequestSelect(event->fX, event->fY);
         PopupContextMenu(fGLViewer->GetSelRec().GetPhysShape(), event, x, y);
      }
      return kTRUE;
   }
   else
   {
      return TEveLegoEventHandler::HandleKey(event);
   }
}
