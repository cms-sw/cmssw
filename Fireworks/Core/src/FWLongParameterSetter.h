#ifndef Fireworks_Core_FWLongParameterSetter_h
#define Fireworks_Core_FWLongParameterSetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWLongParameterSetter
//
/**\class FWLongParameterSetter FWLongParameterSetter.h Fireworks/Core/interface/FWLongParameterSetter.h

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
#include "Fireworks/Core/interface/FWLongParameter.h"

// forward declarations
class TGNumberEntry;

class FWLongParameterSetter : public FWParameterSetterBase {
public:
  FWLongParameterSetter();
  ~FWLongParameterSetter() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  void attach(FWParameterBase*) override;
  TGFrame* build(TGFrame* iParent, bool labelBack = true) override;

  void doUpdate(Long_t);

  FWLongParameterSetter(const FWLongParameterSetter&) = delete;                   // stop default
  const FWLongParameterSetter& operator=(const FWLongParameterSetter&) = delete;  // stop default

private:
  // ---------- member data --------------------------------

  FWLongParameter* m_param;
  TGNumberEntry* m_widget;
};

#endif
