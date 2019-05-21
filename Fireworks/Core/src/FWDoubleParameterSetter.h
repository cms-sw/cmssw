#ifndef Fireworks_Core_FWDoubleParameterSetter_h
#define Fireworks_Core_FWDoubleParameterSetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDoubleParameterSetter
//
/**\class FWDoubleParameterSetter FWDoubleParameterSetter.h Fireworks/Core/interface/FWDoubleParameterSetter.h

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
#include "Fireworks/Core/interface/FWDoubleParameter.h"

// forward declarations
class TGNumberEntry;

class FWDoubleParameterSetter : public FWParameterSetterBase {
public:
  FWDoubleParameterSetter();
  ~FWDoubleParameterSetter() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void attach(FWParameterBase*) override;
  TGFrame* build(TGFrame* iParent, bool labelBack = true) override;

  void setEnabled(bool) override;

  void doUpdate(Long_t);

private:
  FWDoubleParameterSetter(const FWDoubleParameterSetter&) = delete;  // stop default

  const FWDoubleParameterSetter& operator=(const FWDoubleParameterSetter&) = delete;  // stop default

  // ---------- member data --------------------------------
  FWDoubleParameter* m_param;
  TGNumberEntry* m_widget;
};

#endif
