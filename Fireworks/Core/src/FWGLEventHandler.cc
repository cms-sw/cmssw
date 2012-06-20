#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWGUIManager.h"

#include "KeySymbols.h"
#include "TGLViewer.h"
#include "TGLWidget.h"
#include "TGLPhysicalShape.h"
#include "TGLLogicalShape.h"
#include "TEveGedEditor.h"
#include "TEveViewer.h"
#include "FWGeoTopNodeScene.h"

FWGLEventHandler::FWGLEventHandler(TGWindow *w, TObject *obj, TEveCaloLego* l):
   TEveLegoEventHandler(w, obj, l),
   m_viewer(0)
{}

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
            js->GeoPopupMenu(gx, gy, m_viewer->GetGLViewer());
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

Bool_t FWGLEventHandler::HandleFocusChange(Event_t *event)
{
   // Handle generic Event_t type 'event' - provided to catch focus changes
   // and terminate any interaction in viewer.

   if (m_viewer && event->fType == kFocusOut)
      TEveGedEditor::ElementChanged(m_viewer);

   return TGLEventHandler::HandleFocusChange(event);
}

//______________________________________________________________________________
Bool_t FWGLEventHandler::HandleCrossing(Event_t *event)
{
   // Handle generic Event_t type 'event' - provided to catch focus changes
   // and terminate any interaction in viewer.

   if (m_viewer && event->fType == kLeaveNotify)
      TEveGedEditor::ElementChanged(m_viewer);

   return TGLEventHandler::HandleCrossing(event);
}
