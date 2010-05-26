#include "Fireworks/Core/interface/FWGLEventHandler.h"
#define protected public
#include "TGLViewer.h"
#undef protected

#include "TGLWidget.h"
#include "TGWindow.h"
#include "TPoint.h"
#include "TVirtualPad.h" // Remove when pad removed - use signal
#include "TVirtualX.h"
#include "TGClient.h"
#include "TVirtualGL.h"
#include "TGLOverlay.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLAnnotation.h"
#include "TContextMenu.h"
#include "KeySymbols.h"

#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEveProjectionBases.h"

//______________________________________________________________________________
FWGLEventHandler::FWGLEventHandler(TGWindow *w, TObject *obj, TEveCaloLego* l):
   TEveLegoEventHandler(w, obj, l)
{

}

//______________________________________________________________________________
Bool_t FWGLEventHandler::HandleButton(Event_t * event)
{
   // Handle mouse button 'event'.
   static Event_t eventSt = {kOtherEvent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kFALSE, 0, 0, {0,0,0,0,0}};

   // Button DOWN
   if (event->fType == kButtonPress && event->fCode <= kButton3)
   {
      // Allow a single action/button down/up pairing - block others
      fGLViewer->MouseIdle(0, 0, 0);
      fGLViewer->Activated();
      if (fGLViewer->fDragAction != TGLViewer::kDragNone)
         return kFALSE;
      eventSt.fX = event->fX;
      eventSt.fY = event->fY;
      eventSt.fCode = event->fCode;
     
      if ( fGLViewer->GetPushAction() != TGLViewer::kPushStd )
      {
         fGLViewer->RequestSelect(event->fX, event->fY);
         if (fGLViewer->fSelRec.GetN() > 0)
         {
            TGLVector3 v(event->fX, event->fY, 0.5*fGLViewer->fSelRec.GetMinZ());
            fGLViewer->CurrentCamera().WindowToViewport(v);
            v = fGLViewer->CurrentCamera().ViewportToWorld(v);
            if (fGLViewer->GetPushAction() == TGLViewer::kPushCamCenter)
            {
               fGLViewer->CurrentCamera().SetExternalCenter(kTRUE);
               fGLViewer->CurrentCamera().SetCenterVec(v.X(), v.Y(), v.Z());
            }
            else
            {
               TGLSelectRecord& rec = fGLViewer->GetSelRec();
               TObject* obj = rec.GetObject();
               TGLRect& vp = fGLViewer->CurrentCamera().RefViewport();
               TGLAnnotation* ann = new TGLAnnotation(fGLViewer, obj->GetTitle(),  eventSt.fX*1.f/vp.Width(),  1 - eventSt.fY*1.f/vp.Height(), v);
               ann->SetTextSize(0.03);
            }

            fGLViewer->RequestDraw();
         }
         return kTRUE;
      }

      Bool_t grabPointer = kFALSE;
      Bool_t handled     = kFALSE;

      if (fGLViewer->fDragAction == TGLViewer::kDragNone && fGLViewer->fCurrentOvlElm)
      {
         if (fGLViewer->fCurrentOvlElm->Handle(*fGLViewer->fRnrCtx, fGLViewer->fOvlSelRec, event))
         {
            handled     = kTRUE;
            grabPointer = kTRUE;
            fGLViewer->fDragAction = TGLViewer::kDragOverlay;
            fGLViewer->RequestDraw();
         }
      }
      if ( ! handled)
      {
         switch(event->fCode)
         {
            // LEFT mouse button
            case kButton1:
            {
               if (event->fState & kKeyMod1Mask) {
                  fGLViewer->RequestSelect(event->fX, event->fY);
                  fGLViewer->RequestSecondarySelect(event->fX, event->fY);
                  if (fGLViewer->fSecSelRec.GetPhysShape() != 0)
                  {
                     TGLLogicalShape& lshape = const_cast<TGLLogicalShape&>
                        (*fGLViewer->fSecSelRec.GetPhysShape()->GetLogical());
                     lshape.ProcessSelection(*fGLViewer->fRnrCtx, fGLViewer->fSecSelRec);
                     handled = kTRUE;
                  }
               }
               if ( ! handled) {
                  fGLViewer->fDragAction = TGLViewer::kDragCameraRotate;
                  grabPointer = kTRUE;
                  if (fMouseTimer) {
                     fMouseTimer->TurnOff();
                     fMouseTimer->Reset();
                  }
               }
               break;
            }
            // MID mouse button
            case kButton2:
            {
               fGLViewer->fDragAction = TGLViewer::kDragCameraTruck;
               grabPointer = kTRUE;
               break;
            }
            // RIGHT mouse button
            case kButton3:
            {
               if (event->fState & kKeyShiftMask && event->fState & kKeyControlMask )
               {
                  // Root context menu
                  fGLViewer->RequestSelect(event->fX, event->fY);
                  const TGLPhysicalShape * selected = fGLViewer->fSelRec.GetPhysShape();
                  if (selected) { 
                     if (!fGLViewer->fContextMenu) {
                        fGLViewer->fContextMenu = new TContextMenu("glcm", "GL Viewer Context Menu");
                     }
                     Int_t    x, y;
                     Window_t childdum;
                     gVirtualX->TranslateCoordinates(fGLViewer->fGLWidget->GetId(),
                                                     gClient->GetDefaultRoot()->GetId(),
                                                     event->fX, event->fY, x, y, childdum);
                     selected->InvokeContextMenu(*fGLViewer->fContextMenu, x, y);
                  }
               } else {
                  fGLViewer->fDragAction = TGLViewer::kDragCameraDolly;
                  grabPointer = kTRUE;
               }
               break;
            }
         }
      }

      if (grabPointer)
      {
         gVirtualX->GrabPointer(fGLViewer->GetGLWidget()->GetId(),
                                kButtonPressMask | kButtonReleaseMask | kPointerMotionMask,
                                kNone, kNone, kTRUE, kFALSE);
         fInPointerGrab = kTRUE;
      }
   }
   // Button UP
   else if (event->fType == kButtonRelease)
   {
      if (fInPointerGrab)
      {
         gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);
         fInPointerGrab = kFALSE;
      }

      if (fGLViewer->GetPushAction() !=  TGLViewer::kPushStd)
      {
         // This should be 'tool' dependant.
         fGLViewer->fPushAction = TGLViewer::kPushStd;
         return kTRUE;
      }
      else if (fGLViewer->fDragAction == TGLViewer::kDragOverlay && fGLViewer->fCurrentOvlElm)
      {
         fGLViewer->fCurrentOvlElm->Handle(*fGLViewer->fRnrCtx, fGLViewer->fOvlSelRec, event);
         fGLViewer->OverlayDragFinished();
         if (fGLViewer->RequestOverlaySelect(event->fX, event->fY))
            fGLViewer->RequestDraw();
      }
      else if (fGLViewer->fDragAction >= TGLViewer::kDragCameraRotate &&
               fGLViewer->fDragAction <= TGLViewer::kDragCameraDolly)
      {
         fGLViewer->RequestDraw(TGLRnrCtx::kLODHigh);
      }


      switch(event->fCode) {
         // Buttons 4/5 are mouse wheel
         // Note: Modifiers (ctrl/shift) disabled as fState doesn't seem to
         // have correct modifier flags with mouse wheel under Windows.
         case kButton5: {
            // Zoom out (dolly or adjust camera FOV). TODO : val static const somewhere
            if (fGLViewer->CurrentCamera().Zoom(50, kFALSE, kFALSE))
               fGLViewer->fRedrawTimer->RequestDraw(10, TGLRnrCtx::kLODMed);
            return kTRUE;
            break;
         }
         case kButton4: {
            // Zoom in - adjust camera FOV. TODO : val static const somewhere
            if (fGLViewer->CurrentCamera().Zoom(-50, kFALSE, kFALSE))
               fGLViewer->fRedrawTimer->RequestDraw(10, TGLRnrCtx::kLODMed);
            return kTRUE;
            break;
         }
      }
      fGLViewer->fDragAction = TGLViewer::kDragNone;
      if (fGLViewer->fGLDevice != -1)
      {
         gGLManager->MarkForDirectCopy(fGLViewer->fGLDevice, kFALSE);
      }
      if ((event->fX == eventSt.fX) &&
          (event->fY == eventSt.fY) &&
          (eventSt.fCode == event->fCode))
      {
         
         TObject *obj = 0;
         fGLViewer->RequestSelect(fLastPos.fX, fLastPos.fY);
         TGLPhysicalShape *phys_shape = fGLViewer->fSelRec.GetPhysShape();
         
         if (phys_shape) obj = phys_shape->GetLogical()->GetExternal();
         
         // secondary selection
         if (phys_shape && fSecSelType == TGLViewer::kOnRequest
             && phys_shape->GetLogical()->AlwaysSecondarySelect())
         {
            fGLViewer->RequestSecondarySelect(fLastPos.fX, fLastPos.fY);
            fGLViewer->fSecSelRec.SetMultiple(event->fState & kKeyControlMask);
            
            if (fGLViewer->fSecSelRec.GetPhysShape() != 0)
            {
               TGLLogicalShape& lshape = const_cast<TGLLogicalShape&>
               (*fGLViewer->fSecSelRec.GetPhysShape()->GetLogical());
               
               lshape.ProcessSelection(*fGLViewer->fRnrCtx, fGLViewer->fSecSelRec);
               switch (fGLViewer->fSecSelRec.GetSecSelResult())
               {
                  case TGLSelectRecord::kEnteringSelection:
                     fGLViewer->Clicked(obj, event->fCode, event->fState);
                     break;
                  case TGLSelectRecord::kLeavingSelection:
                     fGLViewer->UnClicked(obj, event->fCode, event->fState);
                     break;
                  case TGLSelectRecord::kModifyingInternalSelection:
                     fGLViewer->ReClicked(obj, event->fCode, event->fState);
                     break;
                  default:
                     break;
               };
            }
         }
         else
         {
            fGLViewer->Clicked(obj);
            fGLViewer->Clicked(obj, event->fCode, event->fState);
         }
         
         
         
         eventSt.fX = 0;
         eventSt.fY = 0;
         eventSt.fCode = 0;
         eventSt.fState = 0;

         //Handle context menu
         if(event->fCode ==kButton3) {
            Int_t    x, y;
            Window_t childdum;
            gVirtualX->TranslateCoordinates(fGLViewer->fGLWidget->GetId(),
                                            gClient->GetDefaultRoot()->GetId(),
                                            event->fX, event->fY, x, y, childdum);
            openSelectedModelContextMenu_(x,y);         
         }
         
      }
      if (event->fCode == kButton1 && fMouseTimer)
      {
         fMouseTimer->TurnOn();
      }
   }

   return kTRUE;
}
