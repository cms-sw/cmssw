// -*- C++ -*-
//
// Package:     Core
// Class  :     FWProxyBuilderConfiguration
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:
//         Created:  Wed Jul 27 00:58:43 CEST 2011
//

// system include files

// user include files
#include <iostream>
#include <stdexcept>
#include <functional>

#include "TGFrame.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWItemChangeSignal.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWParameterBase.h"
#include "Fireworks/Core/interface/FWGenericParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"

FWProxyBuilderConfiguration::FWProxyBuilderConfiguration(const FWConfiguration* c, const FWEventItem* item)
    : m_txtConfig(c), m_item(item), m_keepEntries(false) {}

FWProxyBuilderConfiguration::~FWProxyBuilderConfiguration() { delete m_txtConfig; }

//______________________________________________________________________________

void FWProxyBuilderConfiguration::addTo(FWConfiguration& iTo) const {
  if (begin() != end()) {
    FWConfiguration vTmp;
    FWConfigurableParameterizable::addTo(vTmp);
    iTo.addKeyValue("Var", vTmp, true);
  }
}

void FWProxyBuilderConfiguration::setFrom(const FWConfiguration& iFrom) {
  /*
     for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it!= keyVals->end(); ++it)
     std::cout << it->first << "FWProxyBuilderConfiguration::setFrom  " << std::endl;
     }*/
}

//______________________________________________________________________________

void FWProxyBuilderConfiguration::makeSetter(TGCompositeFrame* frame, FWParameterBase* pb) {
  //  std::cout << "make setter " << pb->name() << std::endl;

  std::shared_ptr<FWParameterSetterBase> ptr(FWParameterSetterBase::makeSetterFor(pb));
  ptr->attach(pb, this);
  TGFrame* tmpFrame = ptr->build(frame, false);
  frame->AddFrame(tmpFrame, new TGLayoutHints(kLHintsExpandX));
  m_setters.push_back(ptr);
}

void FWProxyBuilderConfiguration::populateFrame(TGCompositeFrame* settersFrame) {
  //  std::cout << "populate \n";

  TGCompositeFrame* frame = new TGVerticalFrame(settersFrame);
  settersFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX));  //|kLHintsExpandY

  for (const_iterator it = begin(); it != end(); ++it)
    makeSetter(frame, *it);

  settersFrame->MapSubwindows();
}

void FWProxyBuilderConfiguration::keepEntries(bool b) { m_keepEntries = b; }

//______________________________________________________________________________

template <class T>
FWGenericParameter<T>* FWProxyBuilderConfiguration::assertParam(const std::string& name, T def) {
  for (const_iterator i = begin(); i != end(); ++i) {
    if ((*i)->name() == name) {
      return nullptr;
    }
  }

  FWGenericParameter<T>* mode = new FWGenericParameter<T>(this, name, def);

  //   std::cout << "FWProxyBuilderConfiguration::getVarParameter(). No parameter with name " << name << std::endl;
  if (m_txtConfig) {
    const FWConfiguration* varConfig = m_txtConfig->keyValues() ? m_txtConfig->valueForKey("Var") : nullptr;
    if (varConfig)
      mode->setFrom(*varConfig);
  }
  mode->changed_.connect(std::bind(&FWEventItem::proxyConfigChanged, (FWEventItem*)m_item, m_keepEntries));
  return mode;
}

template <class T>
FWGenericParameterWithRange<T>* FWProxyBuilderConfiguration::assertParam(const std::string& name, T def, T min, T max) {
  for (const_iterator i = begin(); i != end(); ++i) {
    if ((*i)->name() == name) {
      return nullptr;
    }
  }

  FWGenericParameterWithRange<T>* mode = new FWGenericParameterWithRange<T>(this, name, def, min, max);

  //   std::cout << "FWProxyBuilderConfiguration::getVarParameter(). No parameter with name " << name << std::endl;
  const FWConfiguration* varConfig =
      m_txtConfig && m_txtConfig->keyValues() ? m_txtConfig->valueForKey("Var") : nullptr;
  if (varConfig)
    mode->setFrom(*varConfig);

  mode->changed_.connect(std::bind(&FWEventItem::proxyConfigChanged, (FWEventItem*)m_item, m_keepEntries));
  return mode;
}

template <class T>
T FWProxyBuilderConfiguration::value(const std::string& pname) {
  FWGenericParameter<T>* param = nullptr;

  for (FWConfigurableParameterizable::const_iterator i = begin(); i != end(); ++i) {
    if ((*i)->name() == pname) {
      param = (FWGenericParameter<T>*)(*i);
      break;
    }
  }

  if (param)
    return param->value();
  else
    throw std::runtime_error("Invalid parameter request.");
}

// explicit template instantiation

template bool FWProxyBuilderConfiguration::value<bool>(const std::string& name);
template long FWProxyBuilderConfiguration::value<long>(const std::string& name);
template double FWProxyBuilderConfiguration::value<double>(const std::string& name);

template FWGenericParameter<bool>* FWProxyBuilderConfiguration::assertParam(const std::string& name, bool def);
template FWGenericParameterWithRange<long>* FWProxyBuilderConfiguration::assertParam(const std::string& name,
                                                                                     long def,
                                                                                     long min,
                                                                                     long max);
template FWGenericParameterWithRange<double>* FWProxyBuilderConfiguration::assertParam(const std::string& name,
                                                                                       double def,
                                                                                       double min,
                                                                                       double max);
