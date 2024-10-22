#ifndef Fireworks_Core_FW3DViewGeometry_h
#define Fireworks_Core_FW3DViewGeometry_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DViewGeometry
//
/**\class FW3DViewGeometry FW3DViewGeometry.h Fireworks/Core/interface/FW3DViewGeometry.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Thu Mar 25 22:06:52 CET 2010
//

#include "Fireworks/Core/interface/FWViewGeometryList.h"

// forward declarations

namespace fireworks {
  class Context;
}

class FW3DViewGeometry : public FWViewGeometryList {
public:
  FW3DViewGeometry(const fireworks::Context& context);
  ~FW3DViewGeometry() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  void showMuonBarrel(bool);
  void showMuonBarrelFull(bool);
  void showMuonEndcap(bool);
  void showPixelBarrel(bool);
  void showPixelEndcap(bool);
  void showTrackerBarrel(bool);
  void showTrackerEndcap(bool);
  void showHGCalEE(bool);
  TEveElementList const* const getHGCalEE() { return m_HGCalEEElements; }
  void showHGCalHSi(bool);
  TEveElementList const* const getHGCalHSi() { return m_HGCalHSiElements; }
  void showHGCalHSc(bool);
  TEveElementList const* const getHGCalHSc() { return m_HGCalHScElements; }
  void showMtdBarrel(bool);
  void showMtdEndcap(bool);

  FW3DViewGeometry(const FW3DViewGeometry&) = delete;  // stop default

  const FW3DViewGeometry& operator=(const FW3DViewGeometry&) = delete;  // stop default

private:
  // ---------- member data --------------------------------

  TEveElementList* m_muonBarrelElements;
  TEveElementList* m_muonBarrelFullElements;
  TEveElementList* m_muonEndcapElements;
  TEveElementList* m_muonEndcapFullElements;
  TEveElementList* m_pixelBarrelElements;
  TEveElementList* m_pixelEndcapElements;
  TEveElementList* m_trackerBarrelElements;
  TEveElementList* m_trackerEndcapElements;
  TEveElementList* m_HGCalEEElements;
  TEveElementList* m_HGCalHSiElements;
  TEveElementList* m_HGCalHScElements;
  TEveElementList* m_mtdBarrelElements;
  TEveElementList* m_mtdEndcapElements;
};

#endif
