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
   TGLAnnotation(view, "vaa", 0.05, 0.95),
   m_event(0)
{
   // Constructor.
   // Create annotation as plain text
   SetRole(TGLOverlayElement::kViewer);
   m_level = new FWLongParameter(confParent, "event info", 1l, 0l, 3l);

   SetUseColorSet(true);
   SetTextSize(0.03);
}

FWEventAnnotation::~FWEventAnnotation()
{
}

//______________________________________________________________________________



void
FWEventAnnotation::setEvent()// const fwlite::Event* event)
{
   m_event = FWGUIManager::getGUIManager()->getCurrentEvent();
   updateText();
}

void
FWEventAnnotation::updateText()
{
   fText = "";

   if (m_event && m_level->value())
   {
      fText += "CMS Experiment at LHC, CERN";
      fText += "\nDate recorded:";
      fText += fw::getLocalTime( *m_event );
      fText += "\nRun/Event: ";
      fText += m_event->id().run();
      fText += "/ ";
      fText += m_event->id().event();
      if ( m_level->value() > 1)
      {
         fText += "\nadd more info for level = 1";
      }
      if ( m_level->value() > 2)
      {
         fText += "\nadd even for more info for level = 2";
      }
   }
   fParent->RequestDraw();
}

void
FWEventAnnotation::Render(TGLRnrCtx& rnrCtx)
{
   if (m_level->value())
      TGLAnnotation::Render(rnrCtx);
}

