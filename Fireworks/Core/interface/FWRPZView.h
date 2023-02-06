#ifndef Fireworks_Core_FWRPZView_h
#define Fireworks_Core_FWRPZView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZView
//
/**\class FWRPZView FWRPZView.h Fireworks/Core/interface/FWRPZView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Feb 19 10:33:21 EST 2008
//

// system include files
#include <string>

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "TEveVector.h"

// forward declarations
class TEveProjectionManager;
class TGLMatrix;
class TEveCalo2D;
class TEveProjectionAxes;
class TEveWindowSlot;
class FWColorManager;
class FWRPZViewGeometry;

class FWRPZView : public FWEveView {
public:
  FWRPZView(TEveWindowSlot* iParent, FWViewType::EType);
  ~FWRPZView() override;

  // ---------- const member functions ---------------------

  void addTo(FWConfiguration&) const override;
  void populateController(ViewerParameterGUI&) const override;
  TEveCaloViz* getEveCalo() const override;

  // ---------- member functions ---------------------------
  void setContext(const fireworks::Context&) override;
  void setFrom(const FWConfiguration&) override;
  void voteCaloMaxVal() override;

  void eventBegin() override;
  void eventEnd() override;
  void setupEventCenter() override;

  //returns the new element created from this import
  void importElements(TEveElement* iProjectableChild, float layer, TEveElement* iProjectedParent = nullptr);

  void shiftOrigin(TEveVector& center);
  void resetOrigin();

  FWRPZView(const FWRPZView&) = delete;                   // stop default
  const FWRPZView& operator=(const FWRPZView&) = delete;  // stop default

private:
  void doPreScaleDistortion();
  void doFishEyeDistortion();
  void doCompression(bool);
  void doShiftOriginToBeamSpot();

  void setEtaRng();

  void showProjectionAxes();
  void projectionAxesLabelSize();

  // ---------- member data --------------------------------
  const static float s_distortF;
  const static float s_distortFInv;

  FWRPZViewGeometry* m_geometryList;
  TEveProjectionManager* m_projMgr;
  TEveProjectionAxes* m_axes;
  TEveCalo2D* m_calo;

  // parameters

  FWBoolParameter m_showPixelBarrel;
  FWBoolParameter m_showPixelEndcap;
  FWBoolParameter m_showTrackerBarrel;
  FWBoolParameter m_showTrackerEndcap;
  FWBoolParameter m_showRpcEndcap;
  FWBoolParameter m_showGEM;
  FWBoolParameter m_showME0;
  FWBoolParameter m_showMtdBarrel;
  FWBoolParameter m_showMtdEndcap;

  FWBoolParameter m_shiftOrigin;
  FWDoubleParameter m_fishEyeDistortion;
  FWDoubleParameter m_fishEyeR;

  FWDoubleParameter m_caloDistortion;
  FWDoubleParameter m_muonDistortion;
  FWBoolParameter m_showProjectionAxes;
  FWDoubleParameter m_projectionAxesLabelSize;
  FWBoolParameter m_compressMuon;

  FWBoolParameter* m_showHF;
  FWBoolParameter* m_showEndcaps;
};

#endif
