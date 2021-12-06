#ifndef Fireworks_Core_FWEveLegoView_h
#define Fireworks_Core_FWEveLegoView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveLegoView
//
/**\class FWEveLegoView FWEveLegoView.h Fireworks/Core/interface/FWEveLegoView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Mon May 31 13:09:38 CEST 2010
//

#include "Fireworks/Core/interface/FWLegoViewBase.h"

class TEveStraightLineSet;

class FWEveLegoView : public FWLegoViewBase {
public:
  FWEveLegoView(TEveWindowSlot*, FWViewType::EType);
  ~FWEveLegoView() override;

  void setContext(const fireworks::Context&) override;
  void setBackgroundColor(Color_t) override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  FWEveLegoView(const FWEveLegoView&) = delete;  // stop default

  const FWEveLegoView& operator=(const FWEveLegoView&) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  TEveStraightLineSet* m_boundaries;
};

#endif
