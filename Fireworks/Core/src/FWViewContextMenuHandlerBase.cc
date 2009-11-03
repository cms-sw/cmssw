// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewContextMenuHandlerBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Mon Nov  2 13:46:48 CST 2009
// $Id: FWViewContextMenuHandlerBase.cc,v 1.1 2009/11/02 23:59:49 chrjones Exp $
//

// system include files
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGLAnnotation.h"

// user include files
#include "Fireworks/Core/interface/FWViewContextMenuHandlerBase.h"
#include "Fireworks/Core/src/FWModelContextMenuHandler.h"


FWViewContextMenuHandlerBase::MenuEntryAdder::MenuEntryAdder(FWModelContextMenuHandler& iHandler):
m_handler(&iHandler),m_lastIndex(0) {}
   
int 
FWViewContextMenuHandlerBase::MenuEntryAdder::addEntry(const char* iEntryName)
{
   m_handler->addViewEntry(iEntryName,m_lastIndex);
   return m_lastIndex++;
}


FWViewContextMenuHandlerBase::FWViewContextMenuHandlerBase()
{
}

FWViewContextMenuHandlerBase::~FWViewContextMenuHandlerBase()
{
}

void 
FWViewContextMenuHandlerBase::addTo(FWModelContextMenuHandler& iHandler)
{
   MenuEntryAdder adder(iHandler);
   init(adder);
}

//==============================================================================
//==============================================================================
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

