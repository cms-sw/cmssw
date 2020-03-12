#include "Fireworks/Candidates/interface/FWLegoCandidate.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//______________________________________________________________________________
FWLegoCandidate::FWLegoCandidate(
    const FWViewContext *vc, const fireworks::Context &context, float energy, float et, float pt, float eta, float phi)
    : m_energy(energy), m_et(et), m_pt(pt), m_eta(eta), m_phi(phi) {
  float base = 0.001;  // Floor offset 1%

  // First vertical line
  FWViewEnergyScale *caloScale = vc->getEnergyScale();
  float val = caloScale->getPlotEt() ? m_pt : m_energy;  // Use pt instead of et

  AddLine(m_eta, m_phi, base, m_eta, m_phi, base + val * caloScale->getScaleFactorLego());

  AddMarker(0, 1.f);
  SetMarkerStyle(3);
  SetMarkerSize(0.01);
  SetDepthTest(false);

  // Circle pt
  const unsigned int nLineSegments = 20;
  const double radius = log(1 + m_pt) / log(10) / 30.f;
  //const double radius = m_pt / 100.f;
  const double twoPi = 2 * TMath::Pi();

  for (unsigned int iPhi = 0; iPhi < nLineSegments; ++iPhi) {
    AddLine(m_eta + radius * cos(twoPi / nLineSegments * iPhi),
            m_phi + radius * sin(twoPi / nLineSegments * iPhi),
            base,
            m_eta + radius * cos(twoPi / nLineSegments * (iPhi + 1)),
            m_phi + radius * sin(twoPi / nLineSegments * (iPhi + 1)),
            base);
  }
}

//______________________________________________________________________________
void FWLegoCandidate::updateScale(const FWViewContext *vc, const fireworks::Context &context) {
  FWViewEnergyScale *caloScale = vc->getEnergyScale();
  float val = caloScale->getPlotEt() ? m_pt : m_energy;  // Use pt instead of et
  float scaleFac = caloScale->getScaleFactorLego();

  // Resize first line
  TEveChunkManager::iterator li(GetLinePlex());
  li.next();
  TEveStraightLineSet::Line_t &l = *(TEveStraightLineSet::Line_t *)li();
  l.fV2[2] = l.fV1[2] + val * scaleFac;

  // Move end point (marker)
  TEveChunkManager::iterator mi(GetMarkerPlex());
  mi.next();
  TEveStraightLineSet::Marker_t &m = *(TEveStraightLineSet::Marker_t *)mi();
  m.fV[2] = l.fV2[2];  // Set to new top of line
}
