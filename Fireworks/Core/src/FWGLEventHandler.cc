#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FW3DViewBase.h"
#include "Fireworks/Core/src/FW3DViewDistanceMeasureTool.h"

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
            js->GeoPopupMenu(gx, gy, m_viewer->viewerGL());
            return;
         }
      }
    
      openSelectedModelContextMenu_(gx,gy);
   }
}

Bool_t FWGLEventHandler::HandleButton(Event_t *event)
{
   Bool_t res = TEveLegoEventHandler::HandleButton(event);
   if (m_viewer->requestGLHandlerPick() && event->fType == kButtonPress )
   {
       Int_t    x, y;
       Window_t childdum;
       gVirtualX->TranslateCoordinates(gClient->GetDefaultRoot()->GetId(), fGLViewer->GetGLWidget()->GetId(),event->fX, event->fY, x, y, childdum);                fGLViewer->RequestSelect(event->fX, event->fY);
       if (fGLViewer->GetSelRec().GetN() > 0 ) 
	 {
	   TGLVector3 v(event->fX, event->fY, 0.5*fGLViewer->GetSelRec().GetMinZ());
	   fGLViewer->CurrentCamera().WindowToViewport(v);
           v = fGLViewer->CurrentCamera().ViewportToWorld(v);
	   FW3DViewBase* v3d = dynamic_cast<FW3DViewBase*>(m_viewer);
	   v3d->setCurrentDMTVertex(v.X(), v.Y(), v.Z());
	 }
   }

   return res;
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
   else {
      return TEveLegoEventHandler::HandleKey(event);

   }
}

Bool_t FWGLEventHandler::HandleFocusChange(Event_t *event)
{
   // Handle generic Event_t type 'event' - provided to catch focus changes
   // and terminate any interaction in viewer.

   if (m_viewer->viewer() && event->fType == kFocusOut)
      TEveGedEditor::ElementChanged(m_viewer->viewer());

   return TGLEventHandler::HandleFocusChange(event);
}

//______________________________________________________________________________
Bool_t FWGLEventHandler::HandleCrossing(Event_t *event)
{
   // Handle generic Event_t type 'event' - provided to catch focus changes
   // and terminate any interaction in viewer.

   if (m_viewer->viewer() && event->fType == kLeaveNotify)
      TEveGedEditor::ElementChanged(m_viewer->viewer());

   return TGLEventHandler::HandleCrossing(event);
}
