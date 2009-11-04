#include "Fireworks/Core/interface/FWViewContextMenuHandlerGL.h"

#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGLAnnotation.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"

void 
FWViewContextMenuHandlerGL::init(FWViewContextMenuHandlerBase::MenuEntryAdder& adder)
{
   adder.addEntry("Add Annotation");
}

void 
FWViewContextMenuHandlerGL::select(int iEntryIndex, const FWModelId &id, int iX, int iY)
{
   if (iEntryIndex == kAnnotate)
   {
      TGFrame* f = m_viewer->GetEveFrame();
      Window_t wdummy;
      Int_t x,y;
      gVirtualX->TranslateCoordinates(gClient->GetDefaultRoot()->GetId(), f->GetId(), iX, iY, x, y, wdummy);

      const char* txt = Form("%s %d", id.item()->name().c_str(), id.index());
      TGLViewer* v = m_viewer->GetGLViewer();
      TGLAnnotation* an = new TGLAnnotation(v, txt,  x*1.f/f->GetWidth(), 1 - y*1.f/f->GetHeight());
      an->SetTextSize(0.035);
      an->SetTransparency(70);
   }
};

