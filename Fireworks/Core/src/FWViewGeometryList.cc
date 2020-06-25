// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewGeometryList
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Tue Sep 14 13:28:13 CEST 2010
//

#include <functional>
#include "TEveScene.h"
#include "TEveManager.h"
#include "TEveCompound.h"
#include "TEveGeoShape.h"
#include "Fireworks/Core/interface/FWViewGeometryList.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/Context.h"

FWViewGeometryList::FWViewGeometryList(const fireworks::Context& context, bool projected)
    : m_context(context), m_geom(nullptr), m_projected(projected) {
  m_geom = context.getGeom();

  for (int i = 0; i < kFWGeomColorSize; ++i) {
    m_colorComp[i] = new TEveCompound(Form("3D view color compund [%d]", i));
    m_colorComp[i]->SetMainColor(m_context.colorManager()->geomColor(FWGeomColorIndex(i)));
    m_colorComp[i]->SetMainTransparency(m_context.colorManager()->geomTransparency(m_projected));
    m_colorComp[i]->CSCApplyMainColorToAllChildren();
    m_colorComp[i]->CSCApplyMainTransparencyToMatchingChildren();
  }
  m_colorConnection =
      context.colorManager()->geomColorsHaveChanged_.connect(std::bind(&FWViewGeometryList::updateColors, this));
  m_transpConnection = context.colorManager()->geomTransparencyHaveChanged_.connect(
      std::bind(&FWViewGeometryList::updateTransparency, this, std::placeholders::_1));
}

FWViewGeometryList::~FWViewGeometryList() {
  m_transpConnection.disconnect();
  m_colorConnection.disconnect();
  for (int i = 0; i < kFWGeomColorSize; ++i) {
    if (m_colorComp[i])
      m_colorComp[i]->Destroy();
  }
}

void FWViewGeometryList::addToCompound(TEveElement* el, FWGeomColorIndex colIdx, bool applyTransp) const {
  el->SetMainColor(m_colorComp[colIdx]->GetMainColor());
  if (applyTransp)
    el->SetMainTransparency(m_colorComp[colIdx]->GetMainTransparency());

  el->SetPickable(true);
  m_colorComp[colIdx]->AddElement(el);
}

void FWViewGeometryList::updateColors() {
  //  printf("%p FWViewGeometryList::updateColors projected %d %s \n", this, m_projected, GetElementName());
  for (int i = 0; i < kFWGeomColorSize; ++i) {
    m_colorComp[i]->SetMainColor(m_context.colorManager()->geomColor(FWGeomColorIndex(i)));
    m_colorComp[i]->SetMainTransparency(m_context.colorManager()->geomTransparency(m_projected));
    m_colorComp[i]->ElementChanged();
  }
}

void FWViewGeometryList::updateTransparency(bool projectedType) {
  //  printf("%p transp [%d]\n", this, iTransp);

  if (projectedType == m_projected) {
    for (int i = 0; i < kFWGeomColorSize; ++i) {
      m_colorComp[i]->SetMainTransparency(m_context.colorManager()->geomTransparency(projectedType));
      m_colorComp[i]->ElementChanged();
    }
  }
}
