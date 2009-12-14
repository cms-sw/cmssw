// -*- C++ -*-
//
// Package:     ESSources
// Class  :     ProxyFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 19:14:11 EDT 2005
//

// system include files

// user include files
#include "CondCore/ESSources/interface/ProxyFactory.h"
#include "CondCore/ESSources/interface/DataProxy.h"

cond::DataProxyWrapperBase::DataProxyWrapperBase(std::string const & il) : m_label(il){}
cond::DataProxyWrapperBase::~DataProxyWrapperBase(){}

EDM_REGISTER_PLUGINFACTORY(cond::ProxyFactory, cond::pluginCategory());

namespace cond {
  const char*
  pluginCategory()
  {
    return  "CondProxyFactory";
  }
}

