#ifndef Fireworks_Core_FWStringParameterSetter_h
#define Fireworks_Core_FWStringParameterSetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWStringParameterSetter
//

// system include files
#include <Rtypes.h>

// user include files
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWStringParameter.h"

// forward declarations
class TGTextEntry;

class FWStringParameterSetter : public FWParameterSetterBase {
public:
  FWStringParameterSetter();
  ~FWStringParameterSetter() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void attach(FWParameterBase*) override;
  TGFrame* build(TGFrame* iParent, bool labelBack = true) override;
  void doUpdate();

private:
  FWStringParameterSetter(const FWStringParameterSetter&) = delete;  // stop default

  const FWStringParameterSetter& operator=(const FWStringParameterSetter&) = delete;  // stop default

  // ---------- member data --------------------------------
  FWStringParameter* m_param;
  TGTextEntry* m_widget;
};

#endif
