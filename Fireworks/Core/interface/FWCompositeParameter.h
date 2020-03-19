#ifndef Fireworks_Core_FWCompositeParameter_h
#define Fireworks_Core_FWCompositeParameter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCompositeParameter
//
/**\class FWCompositeParameter FWCompositeParameter.h Fireworks/Core/interface/FWCompositeParameter.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:37:04 EST 2008
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWParameterBase.h"
#include "Fireworks/Core/interface/FWParameterizable.h"

// forward declarations

class FWCompositeParameter : public FWParameterBase, public FWParameterizable {
public:
  FWCompositeParameter(FWParameterizable* iParent, const std::string& iName, unsigned int iVersion = 1);
  ~FWCompositeParameter() override;

  // ---------- const member functions ---------------------
  void addTo(FWConfiguration&) const override;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void setFrom(const FWConfiguration&) override;

private:
  FWCompositeParameter(const FWCompositeParameter&) = delete;  // stop default

  const FWCompositeParameter& operator=(const FWCompositeParameter&) = delete;  // stop default

  // ---------- member data --------------------------------
  unsigned int m_version;
};

#endif
