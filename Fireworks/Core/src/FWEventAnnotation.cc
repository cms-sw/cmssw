#include "TGLViewer.h"
#include "TEveManager.h"

#include "Fireworks/Core/interface/FWEventAnnotation.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWConfiguration.h"

#include "DataFormats/FWLite/interface/Event.h"

FWEventAnnotation::FWEventAnnotation(TGLViewerBase* view) : TGLAnnotation(view, "Event Info", 0.05, 0.95), m_level(1) {
  SetRole(TGLOverlayElement::kViewer);
  SetUseColorSet(true);
  fAllowClose = false;
}

FWEventAnnotation::~FWEventAnnotation() {}

//______________________________________________________________________________

void FWEventAnnotation::setLevel(long x) {
  if (x != m_level) {
    m_level = x;
    fParent->Changed();
    gEve->Redraw3D();
  }
  updateOverlayText();
}

void FWEventAnnotation::setEvent() { updateOverlayText(); }

void FWEventAnnotation::updateOverlayText() {
  fText = "CMS Experiment at LHC, CERN";

  const edm::EventBase* event = FWGUIManager::getGUIManager()->getCurrentEvent();

  if (event && m_level) {
    fText += "\nData recorded: ";
    fText += fireworks::getLocalTime(*event);
    fText += "\nRun/Event: ";
    fText += event->id().run();
    fText += " / ";
    fText += event->id().event();
    if (m_level > 1) {
      fText += "\nLumi section: ";
      fText += event->luminosityBlock();
    }
    if (m_level > 2) {
      fText += "\nOrbit/Crossing: ";
      fText += event->orbitNumber();
      fText += " / ";
      fText += event->bunchCrossing();
    }
  }

  if (m_level) {
    fParent->Changed();
    gEve->Redraw3D();
  }
}

void FWEventAnnotation::Render(TGLRnrCtx& rnrCtx) {
  if (m_level)
    TGLAnnotation::Render(rnrCtx);
}

//______________________________________________________________________________

void FWEventAnnotation::addTo(FWConfiguration& iTo) const {
  std::stringstream s;
  s << fTextSize;
  iTo.addKeyValue("EventInfoTextSize", FWConfiguration(s.str()));

  std::stringstream x;
  x << fPosX;
  iTo.addKeyValue("EventInfoPosX", FWConfiguration(x.str()));

  std::stringstream y;
  y << fPosY;
  iTo.addKeyValue("EventInfoPosY", FWConfiguration(y.str()));
}

void FWEventAnnotation::setFrom(const FWConfiguration& iFrom) {
  const FWConfiguration* value;

  value = iFrom.valueForKey("EventInfoTextSize");
  if (value)
    fTextSize = atof(value->value().c_str());

  value = iFrom.valueForKey("EventInfoPosX");
  if (value)
    fPosX = atof(value->value().c_str());

  value = iFrom.valueForKey("EventInfoPosY");
  if (value)
    fPosY = atof(value->value().c_str());
}
