#ifndef Fireworks_Core_FWBoolParameterSetter_h
#define Fireworks_Core_FWBoolParameterSetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWBoolParameterSetter
//
/**\class FWBoolParameterSetter FWBoolParameterSetter.h Fireworks/Core/interface/FWBoolParameterSetter.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 10 11:22:26 CDT 2008
//

// system include files
#include <Rtypes.h>

// user include files
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"

// forward declarations
class TGCheckButton;

class FWBoolParameterSetter : public FWParameterSetterBase {
public:
  FWBoolParameterSetter();
  ~FWBoolParameterSetter() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void attach(FWParameterBase*) override;
  TGFrame* build(TGFrame* iParent, bool labelBack = true) override;
  void setEnabled(bool) override;
  void doUpdate();

private:
  FWBoolParameterSetter(const FWBoolParameterSetter&) = delete;  // stop default

  const FWBoolParameterSetter& operator=(const FWBoolParameterSetter&) = delete;  // stop default

  // ---------- member data --------------------------------
  FWBoolParameter* m_param;
  TGCheckButton* m_widget;
};

#endif
