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
//

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "boost/bind.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace {
  void matchLabel(std::pair<std::string, edm::ParameterSetDescription> const& thePair,
                  std::string const& moduleLabel,
                  edm::ParameterSetDescription const*& psetDesc) {
    if (thePair.first == moduleLabel) {
      psetDesc = &thePair.second;
    }
  }
}         

namespace edm {

  ConfigurationDescriptions::ConfigurationDescriptions() :
    unknownDescDefined_(false)
  { }

  ConfigurationDescriptions::~ConfigurationDescriptions() {} 

  void
  ConfigurationDescriptions::setComment(std::string const & value)
  { comment_ = value; }

  void
  ConfigurationDescriptions::setComment(char const* value)
  { comment_ = value; }

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
  ConfigurationDescriptions::validate(ParameterSet & pset,
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

  void ConfigurationDescriptions::print(std::ostream & os,
                                        std::string const& moduleLabel,
                                        bool brief,
                                        bool printOnlyLabels,
                                        size_t lineWidth,
                                        int indentation,
                                        int iPlugin) const {
    if (!brief) {
      if (!comment().empty()) {
        DocFormatHelper::wrapAndPrintText(os, comment(), indentation, lineWidth);
      }
      os << "\n";
    }

    if (descriptions_.empty() && !unknownDescDefined_) {
      indentation += DocFormatHelper::offsetModuleLabel();
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "There are no PSet descriptions defined for this plugin.\n";
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "PSets will not be validated and no cfi files will be generated.\n";
      if (!brief) os << "\n";
      return;
    }

    if (descriptions_.empty() && unknownDescDefined_ && descForUnknownLabels_.isUnknown()) {
      indentation += DocFormatHelper::offsetModuleLabel();
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "This plugin has not implemented the function which defines its\n";
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "configuration descriptions yet. No descriptions are available.\n";
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "Its PSets will not be validated, and no cfi files will be generated.\n";
      if (!brief) os << "\n";
      return;
    }

    if (!brief) {
      std::stringstream ss;
      if (unknownDescDefined_) {
        if (descriptions_.empty()) {
          ss << "This plugin has only one PSet description. "
             << "This description is always used to validate configurations. "
             << "Because this configuration has no module label, no cfi files will be generated.";
        }
        else {
          ss << "This plugin has " << (descriptions_.size() + 1U) << " PSet descriptions. "
             << "The description used to validate a configuration is selected by "
             << "matching the module labels. If none match, then the last description, "
             << "which has no label, is selected. "
             << "A cfi file will be generated for each configuration with a module label.";
        }
      }
      else {
        if (descriptions_.size() == 1U) {
          ss << "This plugin has " << descriptions_.size() << " PSet description. "
             << "This description is always used to validate configurations. "
             << "The module label below is used when generating the cfi file.";
        }
        else {
          ss << "This plugin has " << descriptions_.size() << " PSet descriptions. "
             << "The description used to validate a configuration is selected by "
             << "matching the module labels. If none match the first description below is used. "
             << "The module labels below are also used when generating the cfi files.";
        }
      }
      DocFormatHelper::wrapAndPrintText(os, ss.str(), indentation, lineWidth);
      os << "\n";
    }

    indentation += DocFormatHelper::offsetModuleLabel();

    DescriptionCounter counter;
    counter.iPlugin = iPlugin;
    counter.iSelectedModule = 0;
    counter.iModule = 0;

    for_all(descriptions_, boost::bind(&ConfigurationDescriptions::printForLabel,
                                       this,
                                       _1,
                                       boost::ref(os),
                                       boost::cref(moduleLabel),
                                       brief,
                                       printOnlyLabels,
                                       lineWidth,
                                       indentation,
                                       boost::ref(counter)));

    if (unknownDescDefined_) {
      printForLabel(os,
                    std::string("@default"),
                    descForUnknownLabels_,
                    moduleLabel,
                    brief,
                    printOnlyLabels,
                    lineWidth,
                    indentation,
                    counter);
    }
  }

  void
  ConfigurationDescriptions::printForLabel(std::pair<std::string, ParameterSetDescription> const& labelAndDesc,
                                           std::ostream & os,
                                           std::string const& moduleLabel,
                                           bool brief,
                                           bool printOnlyLabels,
                                           size_t lineWidth,
                                           int indentation,
                                           DescriptionCounter & counter) const
  {
    printForLabel(os,
                  labelAndDesc.first,
                  labelAndDesc.second,
                  moduleLabel,
                  brief,
                  printOnlyLabels,
                  lineWidth,
                  indentation,
                  counter);
  }

  void
  ConfigurationDescriptions::printForLabel(std::ostream & os,
                                           std::string const& label,
                                           ParameterSetDescription const& description,
                                           std::string const& moduleLabel,
                                           bool brief,
                                           bool printOnlyLabels,
                                           size_t lineWidth,
                                           int indentation,
                                           DescriptionCounter & counter) const
  {
    ++counter.iModule;
    if (!moduleLabel.empty() && label != moduleLabel) return;
    ++counter.iSelectedModule;

    std::stringstream ss;
    ss << counter.iPlugin << "." << counter.iSelectedModule;
    std::string section = ss.str();

    os << std::setfill(' ') << std::setw(indentation) << "";
    os << section << " ";
    if (label == std::string("@default")) {
      os << "description without a module label\n";
    }
    else {
      if (!brief) os << "module label: ";
      os << label << "\n";      
    }

    if (!brief) {
      if (!description.comment().empty()) {
        DocFormatHelper::wrapAndPrintText(os, description.comment(), indentation, lineWidth - indentation);        
      }
      os << "\n";
    }
    if (printOnlyLabels) return;

    DocFormatHelper dfh;
    dfh.setBrief(brief);
    dfh.setLineWidth(lineWidth);
    dfh.setIndentation(indentation + DocFormatHelper::offsetTopLevelPSet());
    dfh.setSection(section);
    dfh.setParent(DocFormatHelper::TOP);

    description.print(os, dfh);
  }
}
