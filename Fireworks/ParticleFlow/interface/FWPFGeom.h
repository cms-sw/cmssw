#ifndef _FWPFGEOM_H_
#define _FWPFGEOM_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Namespace:   FWPFGeom
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

//-----------------------------------------------------------------------------
// FWPFGeom
//-----------------------------------------------------------------------------
namespace FWPFGeom {
  // ECAL
  inline float caloR1() { return 129; }  // Centres of front faces of the crystals in supermodules - Barrel
  inline float caloZ1() {
    return 303.353;
  }  // Longitudinal distance between the interaction point and last tracker layer - Endcap

  // HCAL
  inline float caloR2() { return 177.7; }   // Longitudinal profile in the barrel ( inner radius ) - Barrel
  inline float caloR3() { return 287.65; }  // Longitudinal profile in the barrel( outer radius ) - Barrel
  inline float caloZ2() {
    return 400.458;
  }  // Longitudinal distance between the interaction point and the endcap envelope - Endcap
}  // namespace FWPFGeom
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
