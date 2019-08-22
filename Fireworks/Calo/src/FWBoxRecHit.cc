#include "TEveCompound.h"

#include "Fireworks/Calo/interface/FWBoxRecHit.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"

//______________________________________________________________________________
FWBoxRecHit::FWBoxRecHit(const std::vector<TEveVector> &corners, TEveElement *list, float e, float et)
    : m_tower(nullptr), m_ls(nullptr), m_energy(e), m_et(et), m_isTallest(false) {
  buildTower(corners);
  buildLineSet(corners);

  TEveCompound *h = new TEveCompound("rechit box", "tower");
  list->AddElement(h);
  h->CSCApplyMainColorToAllChildren();
  h->AddElement(m_tower);
  h->AddElement(m_ls);
}

//______________________________________________________________________________
/*
 FWViewEnergyScale*
 FWBoxRecHit::getEnergyScale() const
 {
    return  fireworks::Context::getInstance()->commonPrefs()->getEnergyScale();
 }
*/

//______________________________________________________________________________
void FWBoxRecHit::setupEveBox(std::vector<TEveVector> &corners, float scale) {
  // printf("---\n");
  //   TEveVector z(0.f, 0.f, 0.f);
  for (size_t i = 0; i < 4; ++i) {
    int j = i + 4;
    corners[i + 4].fZ = scale;
    m_tower->SetVertex(i, corners[i]);
    m_tower->SetVertex(j, corners[j]);
    //  printf("%ld -> %f, %f , height=%f \n",i, corners[i].fX, corners[i].fY, scale);
  }

  m_tower->SetLineWidth(1.0);
  m_tower->SetLineColor(kBlack);
}

//______________________________________________________________________________
void FWBoxRecHit::buildTower(const std::vector<TEveVector> &corners) {
  m_tower = new TEveBox("EcalRecHitTower");
  std::vector<TEveVector> towerCorners = corners;
  /*
   FWViewEnergyScale *caloScale = 1;//getEnergyScale();
   float val = caloScale->getPlotEt() ? m_et : m_energy;
   float scale = caloScale->getScaleFactorLego() * val;

   if( scale < 0 )
      scale *= -1;
   */
  setupEveBox(towerCorners, 0.01f);
}

//______________________________________________________________________________
void FWBoxRecHit::buildLineSet(const std::vector<TEveVector> &corners) {
  m_ls = new TEveStraightLineSet("EcalRecHitLineSet");

  const float *data;
  TEveVector c;
  for (unsigned int i = 0; i < 4; ++i) {
    data = m_tower->GetVertex(i);
    c.fX += data[0];
    c.fY += data[1];
    m_ls->AddLine(data[0], data[1], 0, data[0], data[1], 0);
  }
  c *= 0.25;

  // last line is trick to add a marker in line set
  m_ls->SetMarkerStyle(1);
  m_ls->AddLine(c.fX, c.fY, c.fZ, c.fX, c.fY, c.fZ);
  m_ls->AddMarker(0, 0.);

  m_ls->ResetBBox();
  m_ls->ComputeBBox();
}

//______________________________________________________________________________
void FWBoxRecHit::updateScale(float scaleFac, float maxLogVal, bool plotEt) {
  //   FWViewEnergyScale *caloScale = getEnergyScale();
  //
  //float scale = caloScale->getScaleFactorLego() * val;

  // printf("scale %f %f\n",  caloScale->getValToHeight(), val);
  float val = plotEt ? m_et : m_energy;
  float scale = scaleFac * val;
  // Reposition top points of tower
  const float *data;
  TEveVector c;
  for (unsigned int i = 0; i < 4; ++i) {
    data = m_tower->GetVertex(i);
    c.fX += data[0];
    c.fY += data[1];
    m_tower->SetVertex(i, data[0], data[1], 0);
    m_tower->SetVertex(i + 4, data[0], data[1], scale);
  }
  c *= 0.25;
  if (false)
    c.Dump();

  // Scale lineset
  float s = log(1 + val) / maxLogVal;
  float d = 0.5 * (m_tower->GetVertex(1)[0] - m_tower->GetVertex(0)[0]);
  d *= s;
  float z = scale * 1.001;
  setLine(0, c.fX - d, c.fY - d, z, c.fX + d, c.fY - d, z);
  setLine(1, c.fX + d, c.fY - d, z, c.fX + d, c.fY + d, z);
  setLine(2, c.fX + d, c.fY + d, z, c.fX - d, c.fY + d, z);
  setLine(3, c.fX - d, c.fY + d, z, c.fX - d, c.fY - d, z);

  if (m_isTallest) {
    m_ls->AddLine(c.fX - d, c.fY - d, z, c.fX + d, c.fY + d, z);
    m_ls->AddLine(c.fX - d, c.fY + d, z, c.fX + d, c.fY - d, z);
    m_ls->GetMarkerPlex().Refit();
  }

  TEveStraightLineSet::Marker_t *m = ((TEveStraightLineSet::Marker_t *)(m_ls->GetMarkerPlex().Atom(0)));
  m->fV[0] = c.fX;
  m->fV[1] = c.fY;
  m->fV[2] = z;

  // stamp changed elements

  m_ls->ComputeBBox();
  //   float* bb = m_ls->GetBBox();
  m_tower->StampTransBBox();
  m_ls->StampTransBBox();
}

//______________________________________________________________________________
void FWBoxRecHit::setLine(int idx, float x1, float y1, float z1, float x2, float y2, float z2) {
  // AMT: this func should go in TEveStraightLineSet class

  TEveStraightLineSet::Line_t *l = ((TEveStraightLineSet::Line_t *)(m_ls->GetLinePlex().Atom(idx)));

  l->fV1[0] = x1;
  l->fV1[1] = y1;
  l->fV1[2] = z1;

  l->fV2[0] = x2;
  l->fV2[1] = y2;
  l->fV2[2] = z2;
}

//______________________________________________________________________________
void FWBoxRecHit::setIsTallest() { m_isTallest = true; }

//______________________________________________________________________________
void FWBoxRecHit::addLine(float x1, float y1, float z1, float x2, float y2, float z2) {
  m_ls->AddLine(x1, y1, z1, x2, y2, z2);
}

//______________________________________________________________________________
void FWBoxRecHit::addLine(const TEveVector &v1, const TEveVector &v2) {
  m_ls->AddLine(v1.fX, v1.fY, v1.fZ, v2.fX, v2.fY, v2.fZ);
}
