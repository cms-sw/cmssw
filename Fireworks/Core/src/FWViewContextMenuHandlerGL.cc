#include "Fireworks/Core/interface/FWViewContextMenuHandlerGL.h"

#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGLAnnotation.h"

void 
FWViewContextMenuHandlerGL::init(FWViewContextMenuHandlerBase::MenuEntryAdder& adder)
{
   adder.addEntry("Add Annotation");
}

void 
FWViewContextMenuHandlerGL::select(int iEntryIndex, int iX, int iY)
{
   if (iEntryIndex == kAnnotate)
   {
      TGFrame* f = m_viewer->GetEveFrame();
      Window_t wdummy;
      Int_t x,y;
      gVirtualX->TranslateCoordinates(gClient->GetDefaultRoot()->GetId(), f->GetId(), iX, iY, x, y, wdummy);

      TGLViewer* v = m_viewer->GetGLViewer();
      TGLAnnotation* an = new TGLAnnotation(v, "Annotate ref == 0",  x*1.f/f->GetWidth(), 1 - y*1.f/f->GetHeight());
      an->SetTextSize(0.05);
      an->SetTransparency(70);
   }
};

