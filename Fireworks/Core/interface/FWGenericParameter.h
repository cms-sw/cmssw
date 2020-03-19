#ifndef Fireworks_Core_FWGenericParameter_h
#define Fireworks_Core_FWGenericParameter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWBoolParameter
//
/**\class FWGenericParameter FWGenericParameter.h Fireworks/Core/interface/FWGenericParameter.h

   Description: Provides access to a simple generic parameter.

   Usage:

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:36:34 EST 2008
//

// system include files
#include <sigc++/signal.h>
#include <sstream>

// user include files
#include "Fireworks/Core/interface/FWParameterBase.h"
#include "Fireworks/Core/interface/FWConfiguration.h"

// forward declarations

template <class T>
class FWGenericParameter : public FWParameterBase {
public:
  typedef T value_type;

  FWGenericParameter() : FWParameterBase(nullptr, "invalid") {}

  FWGenericParameter(FWParameterizable* iParent, const std::string& iName, const T& iDefault = T())
      : FWParameterBase(iParent, iName), m_value(iDefault) {}

  template <class K>
  FWGenericParameter(FWParameterizable* iParent, const std::string& iName, K iCallback, const T& iDefault = T())
      : FWParameterBase(iParent, iName), m_value(iDefault) {
    changed_.connect(iCallback);
  }

  //virtual ~FWBoolParameter();

  // ---------- const member functions ---------------------

  T value() const { return m_value; }

  void addTo(FWConfiguration& iTo) const override {
    std::ostringstream s;
    s << m_value;
    iTo.addKeyValue(name(), FWConfiguration(s.str()));
  }

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  void setFrom(const FWConfiguration& iFrom) override {
    if (const FWConfiguration* config = iFrom.valueForKey(name())) {
      std::istringstream s(config->value());
      s >> m_value;
    }
    changed_(m_value);
  }

  void set(T iValue) {
    m_value = iValue;
    changed_(iValue);
  }

  sigc::signal<void, T> changed_;

private:
  FWGenericParameter(const FWGenericParameter&) = delete;                   // stop default
  const FWGenericParameter& operator=(const FWGenericParameter&) = delete;  // stop default

  // ---------- member data --------------------------------

  T m_value;
};

#endif
