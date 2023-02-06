#ifndef Fireworks_Core_FW3DViewBase_h
#define Fireworks_Core_FW3DViewBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DViewBase
//
/**\class FW3DViewBase FW3DViewBase.h Fireworks/Core/interface/FW3DViewBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
//

// system include files

// user include files
#include "Rtypes.h"
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
// forward declarations
class TEveElementList;
class TEveGeoShape;
class TEveWindowSlot;

class FW3DViewGeometry;
class FWColorManager;
class TGLClip;
class TEveLine;
class TEveBoxSet;

class FW3DViewDistanceMeasureTool;

class FW3DViewBase : public FWEveView {
public:
  FW3DViewBase(TEveWindowSlot*, FWViewType::EType, unsigned int version = 8);
  ~FW3DViewBase() override;

  // ---------- const member functions ---------------------

  void addTo(FWConfiguration&) const override;
  void populateController(ViewerParameterGUI&) const override;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void setContext(const fireworks::Context&) override;
  void setFrom(const FWConfiguration&) override;

  // To be fixed.
  void updateGlobalSceneScaleParameters();

  FW3DViewDistanceMeasureTool* getDMT() { return m_DMT; }
  bool requestGLHandlerPick() const override;
  void setCurrentDMTVertex(double x, double y, double z);

  void showEcalBarrel(bool);

  void setClip(float eta, float phi);

private:
  FW3DViewBase(const FW3DViewBase&);  // stop default

  const FW3DViewBase& operator=(const FW3DViewBase&);  // stop default

  // ---------- member data --------------------------------
  FW3DViewGeometry* m_geometry;
  TGLClip* m_glClip;

  // parameters
  FWEnumParameter m_showMuonBarrel;
  FWBoolParameter m_showMuonEndcap;
  FWBoolParameter m_showPixelBarrel;
  FWBoolParameter m_showPixelEndcap;
  FWBoolParameter m_showTrackerBarrel;
  FWBoolParameter m_showTrackerEndcap;
  FWBoolParameter m_showHGCalEE;
  FWBoolParameter m_showHGCalHSi;
  FWBoolParameter m_showHGCalHSc;
  FWBoolParameter m_showMtdBarrel;
  FWBoolParameter m_showMtdEndcap;

  TEveBoxSet* m_ecalBarrel;
  FWBoolParameter m_showEcalBarrel;

  FWEnumParameter m_rnrStyle;
  FWBoolParameter m_selectable;

  FWEnumParameter m_cameraType;

  FWBoolParameter m_clipEnable;
  FWDoubleParameter m_clipTheta;
  FWDoubleParameter m_clipPhi;
  FWDoubleParameter m_clipDelta1;
  FWDoubleParameter m_clipDelta2;
  FWLongParameter m_clipAppexOffset;
  FWLongParameter m_clipHGCalLayerBegin;
  FWLongParameter m_clipHGCalLayerEnd;

  FW3DViewDistanceMeasureTool* m_DMT;
  TEveLine* m_DMTline;

  void selectable(bool);

  void enableSceneClip(bool);
  void updateClipPlanes(bool resetCamera);
  void updateHGCalVisibility(bool);

  void rnrStyle(long);
  void showMuonBarrel(long);
  void setCameraType(long);
};

#endif
