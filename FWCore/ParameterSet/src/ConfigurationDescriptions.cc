// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ConfigurationDescriptions
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  W. David Dagenhart
//         Created:  17 December 2008
// $Id$
//

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "boost/bind.hpp"

#include <fstream>
#include <iostream>

namespace {
  void matchLabel(std::pair<std::string, edm::ParameterSetDescription> const& thePair,
                 std::string const& moduleLabel,
		  edm::ParameterSetDescription const* psetDesc) {
    if (thePair.first == moduleLabel) {
      psetDesc = &thePair.second;
    }
  }
}         

namespace edm {

  ConfigurationDescriptions::ConfigurationDescriptions() : unknownDescDefined_(false)
  { }

  void
  ConfigurationDescriptions::add(char const* label,
                                 ParameterSetDescription const& psetDescription) {
    std::string labelString(label);
    add(labelString, psetDescription);
  }

  void
  ConfigurationDescriptions::add(std::string const& label,
                                 ParameterSetDescription const& psetDescription) {

    // To minimize the number of copies involved create an empty description first
    // and push it into the vector.  Then perform the copy.
    std::pair<std::string, ParameterSetDescription> pairWithEmptyDescription;
    descriptions_.push_back(pairWithEmptyDescription);
    std::pair<std::string, ParameterSetDescription> & pair = descriptions_.back();

    pair.first = label;
    pair.second = psetDescription;
  }

  void
  ConfigurationDescriptions::addUnknownLabel(ParameterSetDescription const& psetDescription) {
    unknownDescDefined_ = true;
    descForUnknownLabels_ = psetDescription;
  }

  void
  ConfigurationDescriptions::validate(ParameterSet const& pset,
                                      std::string const& moduleLabel) const {
    
    ParameterSetDescription const* psetDesc = 0;
    for_all(descriptions_, boost::bind(&matchLabel,
                                       _1,
                                       boost::cref(moduleLabel),
                                       boost::ref(psetDesc)));

    // If there is a matching label
    if (psetDesc != 0) {
      psetDesc->validate(pset);
    }
    // Is there an explicit description to be used for a non standard label
    else if (unknownDescDefined_) {
      descForUnknownLabels_.validate(pset);
    }
    // Otherwise use the first one.
    else if (descriptions_.size() > 0U) {
      descriptions_[0].second.validate(pset);
    }
    // It is possible for no descriptions to be defined and no validation occurs
    // for this module ever.
  }

  void
  ConfigurationDescriptions::writeCfis(std::string const& baseType,
                                       std::string const& pluginName) const {

    for_all(descriptions_, boost::bind(&ConfigurationDescriptions::writeCfiForLabel,
                                       _1,
                                       boost::cref(baseType),
                                       boost::cref(pluginName)));
  }


  void
  ConfigurationDescriptions::writeCfiForLabel(std::pair<std::string, ParameterSetDescription> const& labelAndDesc,
                                              std::string const& baseType,
                                              std::string const& pluginName)
  {
    std::string cfi_filename = labelAndDesc.first + "_cfi.py";
    std::ofstream outFile(cfi_filename.c_str());


    outFile << "import FWCore.ParameterSet.Config as cms\n\n";
    outFile << labelAndDesc.first << " = cms." << baseType << "('" << pluginName << "'";

    bool startWithComma = true;
    int indentation = 2;
    labelAndDesc.second.writeCfi(outFile, startWithComma, indentation);

    outFile << ")\n";
  
    outFile.close();

    std::cout << labelAndDesc.first << "\n";
  }
}
