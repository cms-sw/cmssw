#include "Fireworks/Core/interface/FWViewContextMenuHandlerGL.h"

#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGLAnnotation.h"
#include "TGLWidget.h"

#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"

FWViewContextMenuHandlerGL::FWViewContextMenuHandlerGL(TEveViewer* v):
m_viewer(v),
m_pickCameraCenter(false)
{
}

void 
FWViewContextMenuHandlerGL::init(FWViewContextMenuHandlerBase::MenuEntryAdder& adder)
{
   adder.addEntry("Add Annotation");
   if (m_pickCameraCenter)
      adder.addEntry("Set Camera Center");
}

void 
FWViewContextMenuHandlerGL::select(int iEntryIndex, const FWModelId &id, int iX, int iY)
{
   TGLViewer* v = m_viewer->GetGLViewer();

   Window_t wdummy;
   Int_t x,y;
   gVirtualX->TranslateCoordinates(gClient->GetDefaultRoot()->GetId(), v->GetGLWidget()->GetId(), iX, iY, x, y, wdummy);   
   TGLVector3 pnt(x, y, 0.5*v->GetSelRec().GetMinZ());
   v->CurrentCamera().WindowToViewport(pnt);
   pnt = v->CurrentCamera().ViewportToWorld(pnt);

   
   if (iEntryIndex == kAnnotate)
   {
      TGFrame* f = v->GetGLWidget();
      Window_t wdummy;
      Int_t x, y;
      gVirtualX->TranslateCoordinates(gClient->GetDefaultRoot()->GetId(), f->GetId(), iX, iY, x, y, wdummy);

      const char* txt = Form("%s %d", id.item()->name().c_str(), id.index());   
      TGLAnnotation* an = new TGLAnnotation(v, txt,  x*1.f/f->GetWidth(), 1 - y*1.f/f->GetHeight(), pnt);
      an->SetUseColorSet(true);
      an->SetTextSize(0.03);
   }
   else if (iEntryIndex == kCameraCenter)
   {
      
      v->CurrentCamera().SetExternalCenter(true);
      v->SetDrawCameraCenter(true);
      v->CurrentCamera().SetCenterVec(pnt.X(), pnt.Y(), pnt.Z());
   }
}
