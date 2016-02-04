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
   {
      adder.addEntry("Set Camera Center");
      adder.addEntry("Reset Camera Center");
   }
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

   switch (iEntryIndex)
   {
      case kAnnotate:
      {
         TGFrame* f = v->GetGLWidget();
         gVirtualX->TranslateCoordinates(gClient->GetDefaultRoot()->GetId(), f->GetId(), iX, iY, x, y, wdummy);

         std::string name = id.item()->modelName(id.index());
         if (id.item()->haveInterestingValue())
            name += ", " + id.item()->modelInterestingValueAsString(id.index());

         TGLAnnotation* an = new TGLAnnotation(v, name.c_str(),  x*1.f/f->GetWidth(), 1 - y*1.f/f->GetHeight(), pnt);
         an->SetUseColorSet(true);
         an->SetTextSize(0.03);
         break;
      }
      case kCameraCenter:
      {
         v->CurrentCamera().SetExternalCenter(true);
         v->SetDrawCameraCenter(true);
         v->CurrentCamera().SetCenterVec(pnt.X(), pnt.Y(), pnt.Z());
         break;
      }
      case kResetCameraCenter:
      {
         v->CurrentCamera().SetExternalCenter(false);
         v->SetDrawCameraCenter(false);
         break;
      }
   }
}
