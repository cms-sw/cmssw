#ifndef FWCore_ParameterSet_PluginDescriptionAdaptor_h
#define FWCore_ParameterSet_PluginDescriptionAdaptor_h
// -*- C++ -*-
//
// Package:     FWCore/ParameterSet
// Class  :     PluginDescriptionAdaptor
// 
/**\class PluginDescriptionAdaptor PluginDescriptionAdaptor.h "PluginDescriptionAdaptor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 19 Sep 2018 19:24:28 GMT
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/PluginDescriptionAdaptorBase.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations

namespace edm {

template<typename BASE, typename T>
  class PluginDescriptionAdaptor : public PluginDescriptionAdaptorBase<BASE>
{
public:
  // ---------- const member functions ---------------------
  edm::ParameterSetDescription description() const final {
    edm::ParameterSetDescription d;
    T::fillPSetDescription(d);
    return d;
  }
};
}
#endif
