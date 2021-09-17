#ifndef _FWBOXRECHIT_H_
#define _FWBOXRECHIT_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWBoxRecHit
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include "TEveBox.h"
#include "TEveCompound.h"
#include "TEveStraightLineSet.h"

#include "TEveCaloData.h"
#include "TEveChunkManager.h"

// User include files
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/Context.h"

// Forward declarations
class FWProxyBuilderBase;

//-----------------------------------------------------------------------------
// FWBoxRechHit
//-----------------------------------------------------------------------------
class FWBoxRecHit {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWBoxRecHit(const std::vector<TEveVector> &corners, TEveElement *comp, float e, float et);
  virtual ~FWBoxRecHit() {}

  // --------------------- Member Functions --------------------------
  void updateScale(float scale, float maxLogVal, bool plotEt);
  void setSquareColor(Color_t c) {
    m_ls->SetMarkerColor(c);
    m_ls->SetLineColor(kBlack);
  }

  TEveBox *getTower() { return m_tower; }
  void setLine(int idx, float x1, float y1, float z1, float x2, float y2, float z2);
  void addLine(float x1, float y1, float z1, float x2, float y2, float z2);
  void addLine(const TEveVector &v1, const TEveVector &v2);
  float getEnergy(bool b) const { return b ? m_et : m_energy; }
  bool isTallest() const { return m_isTallest; }
  void setIsTallest();

  FWBoxRecHit(const FWBoxRecHit &) = delete;                   // Disable default
  const FWBoxRecHit &operator=(const FWBoxRecHit &) = delete;  // Disable default

private:
  // --------------------- Member Functions --------------------------
  void setupEveBox(std::vector<TEveVector> &corners, float scale);
  void buildTower(const std::vector<TEveVector> &corners);
  void buildLineSet(const std::vector<TEveVector> &corners);

  // ----------------------- Data Members ----------------------------
  TEveBox *m_tower;
  TEveStraightLineSet *m_ls;
  float m_energy;
  float m_et;
  bool m_isTallest;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
