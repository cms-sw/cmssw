#ifndef FWCore_ParameterSet_ConfigurationDescriptions_h
#define FWCore_ParameterSet_ConfigurationDescriptions_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ConfigurationDescriptions
// 
/**\class ConfigurationDescriptions ConfigurationDescriptions.h FWCore/ParameterSet/interface/ConfigurationDescriptions.h

 Description: Used to hold ParameterSetDescriptions with corresponding module labels

 Usage:
    <usage>

*/
//
// Original Author:  W. David Dagenhart
//         Created:  17 December 2008
// $Id$
//

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>
#include <string>
#include <utility>

namespace edm {

  class ConfigurationDescriptions {
  public:

    ConfigurationDescriptions();

    void add(std::string const& label, ParameterSetDescription const& psetDescription);

    void add(char const* label, ParameterSetDescription const& psetDescription);

    void addUnknownLabel(ParameterSetDescription const& psetDescription);

    void validate(ParameterSet const& pset, std::string const& moduleLabel) const;

    void writeCfis(std::string const& baseType,
                   std::string const& pluginName) const;

  private:

    static void writeCfiForLabel(std::pair<std::string, ParameterSetDescription> const& labelAndDesc,
                                 std::string const& baseType,
                                 std::string const& pluginName);

    std::vector<std::pair<std::string, ParameterSetDescription> > descriptions_;

    bool unknownDescDefined_;
    ParameterSetDescription descForUnknownLabels_;
  };
}

#endif
