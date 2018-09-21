#ifndef FWCore_ParameterSet_PluginDescriptionAdaptorBase_h
#define FWCore_ParameterSet_PluginDescriptionAdaptorBase_h
// -*- C++ -*-
//
// Package:     FWCore/ParameterSet
// Class  :     PluginDescriptionAdaptorBase
// 
/**\class PluginDescriptionAdaptorBase PluginDescriptionAdaptorBase.h "PluginDescriptionAdaptorBase.h"

 Description: Base class for the adaptor used to call fillPSetDescription of a plugin

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 19 Sep 2018 19:24:24 GMT
//

// system include files
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// user include files

// forward declarations

namespace edm {
template<typename T>
class PluginDescriptionAdaptorBase
{
public:
  virtual ~PluginDescriptionAdaptorBase() = default;
  
  // ---------- const member functions ---------------------
  virtual edm::ParameterSetDescription description() const = 0;
};
}
#endif
