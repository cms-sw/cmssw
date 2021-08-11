#ifndef Fireworks_Core_FWGlimpseView_h
#define Fireworks_Core_FWGlimpseView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseView
//
/**\class FWGlimpseView FWGlimpseView.h Fireworks/Core/interface/FWGlimpseView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"

// forward declarations
class TEveViewer;
class TEveScene;
class TEveElementList;
class TEveWindowSlot;
class TEveGeoShape;

class FWGlimpseView : public FWEveView {
public:
  FWGlimpseView(TEveWindowSlot*, FWViewType::EType);
  ~FWGlimpseView() override;

  // ---------- const member functions ---------------------

  void addTo(FWConfiguration&) const override;
  void setFrom(const FWConfiguration&) override;

  // ---------- static member functions --------------------

  FWGlimpseView(const FWGlimpseView&) = delete;                   // stop default
  const FWGlimpseView& operator=(const FWGlimpseView&) = delete;  // stop default

private:
  void createAxis();
  void showAxes();
  void showCylinder();

  // ---------- member data --------------------------------
  TEveGeoShape* m_cylinder;

  // FWDoubleParameter m_scaleParam;
  FWBoolParameter m_showAxes;
  FWBoolParameter m_showCylinder;
};

#endif
