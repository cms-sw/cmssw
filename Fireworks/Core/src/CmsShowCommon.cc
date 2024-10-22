// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowCommon
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Sep 10 14:50:32 CEST 2010
//

// system include files
#include <functional>
#include <iostream>
// user include files

#include "TEveManager.h"
#include "TEveTrackPropagator.h"
#include "TGLViewer.h"
#include "TEveViewer.h"
#include "TGComboBox.h"
#include "TGClient.h"
#include "TGTextEntry.h"

#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/CmsShowCommonPopup.h"
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//
// constructors and destructor
//
CmsShowCommon::CmsShowCommon(fireworks::Context* c)
    : FWConfigurableParameterizable(2),
      m_view(nullptr),
      m_context(c),
      m_trackBreak(this, "     ", 2l, 0l, 2l),  // do not want to render text at setter
      m_drawBreakPoints(this, "Show y=0 points as markers", false),
      m_backgroundColor(this, "backgroundColIdx", 1l, 0l, 1000l),
      m_gamma(this, "Brightness", 0l, -15l, 15l),
      m_palette(this, "Palette", 1l, 0l, 2l),
      m_geomTransparency2D(this, "Transparency 2D", long(colorManager()->geomTransparency(true)), 0l, 100l),
      m_geomTransparency3D(this, "Transparency 3D", long(colorManager()->geomTransparency(false)), 0l, 100l),
      m_useBeamSpot(true) {
  m_viewContext.setEnergyScale(new FWViewEnergyScale("global", 2));
  // projections
  m_trackBreak.addEntry(0, "Jump to proper hemisphere");
  m_trackBreak.addEntry(1, "Stay on first point side");
  m_trackBreak.addEntry(2, "Stay on last point side");

  m_palette.addEntry(FWColorManager::kClassic, "Classic ");
  m_palette.addEntry(FWColorManager::kPurple, "Purple ");
  m_palette.addEntry(FWColorManager::kFall, "Fall ");
  m_palette.addEntry(FWColorManager::kSpring, "Spring ");
  m_palette.addEntry(FWColorManager::kArctic, "Arctic ");

  // colors
  char name[32];
  for (int i = 0; i < kFWGeomColorSize; ++i) {
    snprintf(name, 31, "GeometryColor %d ", i);
    m_geomColors[i] =
        new FWLongParameter(this, name, long(colorManager()->geomColor(FWGeomColorIndex(i))), 1000l, 1100l);
  }

  m_trackBreak.changed_.connect(std::bind(&CmsShowCommon::setTrackBreakMode, this));
  m_palette.set(m_context->colorManager()->getPalette());
  m_drawBreakPoints.changed_.connect(std::bind(&CmsShowCommon::setDrawBreakMarkers, this));
  m_gamma.changed_.connect(std::bind(&CmsShowCommon::setGamma, this));

  m_lightColorSet.StdLightBackground();
  m_darkColorSet.StdDarkBackground();
}

CmsShowCommon::~CmsShowCommon() {}

const FWColorManager* CmsShowCommon::colorManager() const { return m_context->colorManager(); }
//
// member functions
//

void CmsShowCommon::setTrackBreakMode() {
  if (m_context->getTrackPropagator()->GetProjTrackBreaking() != m_trackBreak.value()) {
    m_context->getTrackPropagator()->SetProjTrackBreaking(m_trackBreak.value());
    m_context->getTrackerTrackPropagator()->SetProjTrackBreaking(m_trackBreak.value());
    m_context->getMuonTrackPropagator()->SetProjTrackBreaking(m_trackBreak.value());
    gEve->Redraw3D();
  }
}

void CmsShowCommon::setDrawBreakMarkers() {
  if (m_context->getTrackPropagator()->GetRnrPTBMarkers() != m_drawBreakPoints.value()) {
    m_context->getTrackPropagator()->SetRnrPTBMarkers(m_drawBreakPoints.value());
    m_context->getTrackerTrackPropagator()->SetRnrPTBMarkers(m_drawBreakPoints.value());
    m_context->getMuonTrackPropagator()->SetRnrPTBMarkers(m_drawBreakPoints.value());
    gEve->Redraw3D();
  }
}

void CmsShowCommon::setGamma() { m_context->colorManager()->setBrightness(m_gamma.value()); }

void CmsShowCommon::switchBackground() {
  m_context->colorManager()->switchBackground();
  m_backgroundColor.set(colorManager()->background());
}

void CmsShowCommon::permuteColors() {
  // printf("Reverting order of existing colors ...\n");

  std::vector<Color_t> colv;
  colv.reserve(64);

  for (FWEventItemsManager::const_iterator i = m_context->eventItemsManager()->begin();
       i != m_context->eventItemsManager()->end();
       ++i) {
    colv.push_back((*i)->defaultDisplayProperties().color());
  }

  int vi = colv.size() - 1;
  for (FWEventItemsManager::const_iterator i = m_context->eventItemsManager()->begin();
       i != m_context->eventItemsManager()->end();
       ++i, --vi) {
    FWDisplayProperties prop = (*i)->defaultDisplayProperties();
    prop.setColor(colv[vi]);
    (*i)->setDefaultDisplayProperties(prop);

    (*i)->defaultDisplayPropertiesChanged_(*i);
  }
}

void CmsShowCommon::randomizeColors() {
  //   printf("Doing random_shuffle on existing colors ...\n");

  int vi = 0;
  for (FWEventItemsManager::const_iterator i = m_context->eventItemsManager()->begin();
       i != m_context->eventItemsManager()->end();
       ++i, ++vi) {
    FWDisplayProperties prop = (*i)->defaultDisplayProperties();

    int col = rand() % 17;  // randomize in first row of palette
    prop.setColor(col);
    (*i)->setDefaultDisplayProperties(prop);

    (*i)->defaultDisplayPropertiesChanged_(*i);
  }
}

void CmsShowCommon::setGeomColor(FWGeomColorIndex cidx, Color_t iColor) {
  m_geomColors[cidx]->set(iColor);
  m_context->colorManager()->setGeomColor(cidx, iColor);
}

void CmsShowCommon::setGeomTransparency(int iTransp, bool projected) {
  if (projected)
    m_geomTransparency2D.set(iTransp);
  else
    m_geomTransparency3D.set(iTransp);

  m_context->colorManager()->setGeomTransparency(iTransp, projected);
}

//____________________________________________________________________________

namespace {
  void addGLColorToConfig(const char* cname, const TGLColor& c, FWConfiguration& oTo) {
    FWConfiguration pc;

    std::ostringstream sRed;
    sRed << (int)c.GetRed();
    pc.addKeyValue("Red", sRed.str());

    std::ostringstream sGreen;
    sGreen << (int)c.GetGreen();
    pc.addKeyValue("Green", sGreen.str());

    std::ostringstream sBlue;
    sBlue << (int)c.GetBlue();
    pc.addKeyValue("Blue", sBlue.str());

    oTo.addKeyValue(cname, pc, true);
  }

  void setGLColorFromConfig(TGLColor& d, const FWConfiguration* iFrom) {
    if (!iFrom)
      return;
    d.Arr()[0] = atoi(iFrom->valueForKey("Red")->value().c_str());
    d.Arr()[1] = atoi(iFrom->valueForKey("Green")->value().c_str());
    d.Arr()[2] = atoi(iFrom->valueForKey("Blue")->value().c_str());
    //    printf("22222 colors %d %d %d \n",  d.Arr()[0],  d.Arr()[1], d.Arr()[2]);
  }
}  // namespace

void CmsShowCommon::addTo(FWConfiguration& oTo) const {
  m_backgroundColor.set(int(colorManager()->background()));

  FWConfigurableParameterizable::addTo(oTo);
  m_viewContext.getEnergyScale()->addTo(oTo);

  if (gEve) {
    addGLColorToConfig("SelectionColorLight", m_lightColorSet.Selection(1), oTo);
    addGLColorToConfig("HighlightColorLight", m_lightColorSet.Selection(3), oTo);
    addGLColorToConfig("SelectionColorDark", m_darkColorSet.Selection(1), oTo);
    addGLColorToConfig("HighlightColorDark", m_darkColorSet.Selection(3), oTo);
  }
}

void CmsShowCommon::setFrom(const FWConfiguration& iFrom) {
  for (const_iterator it = begin(), itEnd = end(); it != itEnd; ++it) {
    (*it)->setFrom(iFrom);
  }

  if (iFrom.valueForKey("Palette"))
    setPalette();

  // handle old and new energy scale configuration if existing
  if (iFrom.valueForKey("ScaleMode")) {
    long mode = atol(iFrom.valueForKey("ScaleMode")->value().c_str());

    float convert;
    if (iFrom.valueForKey("EnergyToLength [GeV/m]"))
      convert = atof(iFrom.valueForKey("EnergyToLength [GeV/m]")->value().c_str());
    else
      convert = atof(iFrom.valueForKey("ValueToHeight [GeV/m]")->value().c_str());

    float maxH;
    if (iFrom.valueForKey("MaximumLength [m]"))
      maxH = atof(iFrom.valueForKey("MaximumLength [m]")->value().c_str());
    else
      maxH = atof(iFrom.valueForKey("MaxTowerH [m]")->value().c_str());

    int et = atoi(iFrom.valueForKey("PlotEt")->value().c_str());
    m_viewContext.getEnergyScale()->SetFromCmsShowCommonConfig(mode, convert, maxH, et);
  }

  // background
  FWColorManager* cm = m_context->colorManager();
  cm->setBackgroundAndBrightness(FWColorManager::BackgroundColorIndex(m_backgroundColor.value()), m_gamma.value());

  // geom colors
  cm->setGeomTransparency(m_geomTransparency2D.value(), true);
  cm->setGeomTransparency(m_geomTransparency3D.value(), false);

  for (int i = 0; i < kFWGeomColorSize; ++i)
    cm->setGeomColor(FWGeomColorIndex(i), m_geomColors[i]->value());

  if (gEve) {
    setGLColorFromConfig(m_lightColorSet.Selection(1), iFrom.valueForKey("SelectionColorLight"));
    setGLColorFromConfig(m_lightColorSet.Selection(3), iFrom.valueForKey("HighlightColorLight"));
    setGLColorFromConfig(m_darkColorSet.Selection(1), iFrom.valueForKey("SelectionColorDark"));
    setGLColorFromConfig(m_darkColorSet.Selection(3), iFrom.valueForKey("HighlightColorDark"));
  }
}

void CmsShowCommon::setPalette() {
  FWColorManager* cm = m_context->colorManager();
  cm->setPalette(m_palette.value());

  for (int i = 0; i < kFWGeomColorSize; ++i) {
    FWColorManager* cm = m_context->colorManager();
    m_geomColors[i]->set(cm->geomColor(FWGeomColorIndex(i)));
    //      m_colorSelectWidget[i]->SetColorByIndex(cm->geomColor(FWGeomColorIndex(i)), kFALSE);
  }

  m_context->colorManager()->propagatePaletteChanges();
  for (FWEventItemsManager::const_iterator i = m_context->eventItemsManager()->begin();
       i != m_context->eventItemsManager()->end();
       ++i) {
    (*i)->resetColor();
  }
}

void CmsShowCommon::loopPalettes() {
  int val = m_palette.value();
  val++;

  if (val >= FWColorManager::kPaletteLast)
    val = FWColorManager::kPaletteFirst;

  if (m_view) {
    TGComboBox* combo = m_view->getCombo();
    combo->Select(val, true);
  } else {
    m_palette.set(val);
    setPalette();
  }
}

void CmsShowCommon::getEventCenter(float* iC) const {
  if (m_useBeamSpot) {
    FWBeamSpot* beamSpot = fireworks::Context::getInstance()->getBeamSpot();
    iC[0] = float(beamSpot->x0());
    iC[1] = float(beamSpot->y0());
    iC[2] = float(beamSpot->z0());
  } else {
    iC[0] = m_externalEventCenter.fX;
    iC[1] = m_externalEventCenter.fY;
    iC[2] = m_externalEventCenter.fZ;
  }
}

void CmsShowCommon::setEventCenter(float x, float y, float z) {
  m_useBeamSpot = false;
  m_externalEventCenter.Set(x, y, z);
  eventCenterChanged_.emit(this);
}

void CmsShowCommon::resetEventCenter() {
  if (!m_useBeamSpot) {
    fwLog(fwlog::kInfo) << "CmsShowCommon re-set event center to BeamSpot\n ";
    m_useBeamSpot = true;
    FWBeamSpot* beamSpot = fireworks::Context::getInstance()->getBeamSpot();
    setEventCenter(beamSpot->x0(), beamSpot->y0(), beamSpot->z0());
  }
}
