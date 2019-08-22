#include "Fireworks/ParticleFlow/interface/FWPFRhoPhiRecHit.h"

//______________________________________________________________________________
FWPFRhoPhiRecHit::FWPFRhoPhiRecHit(FWProxyBuilderBase *pb,
                                   TEveElement *iH,
                                   const FWViewContext *vc,
                                   float E,
                                   float et,
                                   double lPhi,
                                   double rPhi,
                                   std::vector<TEveVector> &bCorners)
    : m_hasChild(false), m_energy(E), m_et(et), m_lPhi(lPhi), m_rPhi(rPhi), m_child(nullptr) {
  buildRecHit(pb, iH, vc, bCorners);
}

//______________________________________________________________________________
FWPFRhoPhiRecHit::~FWPFRhoPhiRecHit() {}

//______________________________________________________________________________
void FWPFRhoPhiRecHit::updateScale(const FWViewContext *vc) {
  FWViewEnergyScale *caloScale = vc->getEnergyScale();
  float value = caloScale->getPlotEt() ? m_et : m_energy;
  Double_t scale = caloScale->getScaleFactor3D() * value;
  unsigned int a = 0;

  if (scale < 0.f)
    scale *= -1.f;

  // Scale centres
  TEveVector sc1 = m_corners[1];
  TEveVector sc2 = m_corners[0];

  // Used to store normalized vectors
  TEveVector v1 = sc1;  // Bottom right corner
  TEveVector v2 = sc2;  // Bottom left corner

  v1.Normalize();
  v2.Normalize();

  v1 *= scale;  // Now at new height
  v2 *= scale;

  // Get line parameters and scale coordinates
  TEveChunkManager::iterator li(m_ls->GetLinePlex());
  while (li.next()) {
    TEveStraightLineSet::Line_t &l = *(TEveStraightLineSet::Line_t *)li();
    switch (a) {
      case 0:
        // Left side of tower first
        l.fV1[0] = sc2.fX;
        l.fV1[1] = sc2.fY;
        l.fV2[0] = sc2.fX + v2.fX;
        l.fV2[1] = sc2.fY + v2.fY;
        break;

      case 1:
        // Top of tower
        l.fV1[0] = sc2.fX + v2.fX;
        l.fV1[1] = sc2.fY + v2.fY;
        l.fV2[0] = sc1.fX + v1.fX;
        l.fV2[1] = sc1.fY + v1.fY;
        break;

      case 2:
        // Right hand side of tower
        l.fV1[0] = sc1.fX + v1.fX;
        l.fV1[1] = sc1.fY + v1.fY;
        l.fV2[0] = sc1.fX;
        l.fV2[1] = sc1.fY;
        break;

      case 3:
        // Bottom of tower
        l.fV1[0] = sc1.fX;
        l.fV1[1] = sc1.fY;
        l.fV2[0] = sc2.fX;
        l.fV2[1] = sc2.fY;
        break;
    }
    a++;
  }
  TEveProjected *proj = *(m_ls)->BeginProjecteds();
  proj->UpdateProjection();

  m_corners[2] = sc2 + v2;  // New top left of tower
  m_corners[3] = sc1 + v1;  // New top right of tower

  if (m_hasChild) {
    m_child->setCorners(0, m_corners[2]);
    m_child->setCorners(1, m_corners[3]);  // Base of child is now top of parent
    m_child->updateScale(vc);
  }
}

//______________________________________________________________________________
void FWPFRhoPhiRecHit::clean() {
  m_corners.clear();
  if (m_hasChild)
    m_child->clean();

  delete this;
}

//______________________________________________________________________________
void FWPFRhoPhiRecHit::addChild(
    FWProxyBuilderBase *pb, TEveElement *itemHolder, const FWViewContext *vc, float E, float et) {
  if (m_hasChild)  // Already has a child stacked on top so move on to child
    m_child->addChild(pb, itemHolder, vc, E, et);
  else {
    std::vector<TEveVector> corners(2);
    corners[0] = m_corners[2];  // Top left of current tower
    corners[1] = m_corners[3];  // Top right of current tower
    m_child = new FWPFRhoPhiRecHit(pb, itemHolder, vc, E, et, m_lPhi, m_rPhi, corners);
    m_hasChild = true;
  }
}

//______________________________________________________________________________
void FWPFRhoPhiRecHit::buildRecHit(FWProxyBuilderBase *pb,
                                   TEveElement *itemHolder,
                                   const FWViewContext *vc,
                                   std::vector<TEveVector> &bCorners) {
  float scale = 0;
  float value = 0;
  TEveVector v1, v2, v3, v4;
  TEveVector vec;

  FWViewEnergyScale *caloScale = vc->getEnergyScale();
  value = caloScale->getPlotEt() ? m_et : m_energy;
  scale = caloScale->getScaleFactor3D() * value;

  v1 = bCorners[0];  // Bottom left
  v2 = bCorners[1];  // Bottom right

  v3 = v1;
  vec = v3;
  vec.Normalize();
  v3 = v3 + (vec * scale);

  v4 = v2;
  vec = v4;
  vec.Normalize();
  v4 = v4 + (vec * scale);

  m_ls = new TEveScalableStraightLineSet("rhophiRecHit");
  m_ls->AddLine(v1.fX, v1.fY, 0, v3.fX, v3.fY, 0);  // Bottom left - Top left
  m_ls->AddLine(v3.fX, v3.fY, 0, v4.fX, v4.fY, 0);  // Top left - Top right
  m_ls->AddLine(v4.fX, v4.fY, 0, v2.fX, v2.fY, 0);  // Top right - Bottom right
  m_ls->AddLine(v2.fX, v2.fY, 0, v1.fX, v1.fY, 0);  // Bottom right - Bottom left

  m_corners.push_back(v1);
  m_corners.push_back(v2);
  m_corners.push_back(v3);
  m_corners.push_back(v4);

  pb->setupAddElement(m_ls, itemHolder);
}
