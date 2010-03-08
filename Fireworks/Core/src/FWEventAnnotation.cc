#include "TGLIncludes.h"
#include "TROOT.h"
#include "TColor.h"
#include "TGLUtil.h"
#include "TGLCamera.h"
#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLViewerBase.h"
#include "TObjString.h"
#include "TGLViewer.h"
#include "TMath.h"
#include "TImage.h"
#include <KeySymbols.h>

#include "Fireworks/Core/interface/FWEventAnnotation.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/FWLite/interface/Event.h"

FWEventAnnotation::FWEventAnnotation(TGLViewerBase *view,  FWParameterizable* confParent):
   TGLAnnotation(view, "Event Info", 0.05, 0.95),
   m_event(0),
   m_level(1)
{
   SetRole(TGLOverlayElement::kViewer);

   SetUseColorSet(true);
   SetTextSize(0.03);
}

FWEventAnnotation::~FWEventAnnotation()
{
}

//______________________________________________________________________________

void
FWEventAnnotation::setLevel(long x)
{ 
   m_level = x;
   updateOverlayText();
}


void
FWEventAnnotation::setEvent()
{
   m_event = FWGUIManager::getGUIManager()->getCurrentEvent();
   updateOverlayText();
}

void
FWEventAnnotation::updateOverlayText()
{
   fText = "CMS Experiment at LHC, CERN";

   if (m_event && m_level)
   {
      fText += "\nDate recorded:";
      fText += fw::getLocalTime( *m_event );
      fText += "\nRun/Event: ";
      fText += m_event->id().run();
      fText += "/ ";
      fText += m_event->id().event();
      if ( m_level > 1)
      {
         fText += "\nadd more info for level = 2";
      }
      if ( m_level > 2)
      {
         fText += "\nadd even for more info for level = 3";
      }
   }
   fParent->RequestDraw();
}

void
FWEventAnnotation::Render(TGLRnrCtx& rnrCtx)
{
   if (m_level)
      TGLAnnotation::Render(rnrCtx);
}

