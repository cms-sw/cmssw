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
#include "FWCore/Utilities/interface/EDMException.h"

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

  ConfigurationDescriptions::ConfigurationDescriptions(std::string const& baseType) :
    baseType_(baseType),
    defaultDescDefined_(false)
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

    if (baseType_ == std::string("Source")) {
      if (label != std::string("source")) {
        throw edm::Exception(edm::errors::LogicError,
          "ConfigurationDescriptions::add, when adding a ParameterSetDescription for a source the label must be \"source\"\n");
      }
      if (descriptions_.size() != 0U ||
          defaultDescDefined_ == true) {
        throw edm::Exception(edm::errors::LogicError,
          "ConfigurationDescriptions::add, for a source only 1 ParameterSetDescription may be added\n");
      }
    }
    else if (baseType_ == std::string("Service")) {
      if (descriptions_.size() != 0U ||
          defaultDescDefined_ == true) {
        throw edm::Exception(edm::errors::LogicError,
          "ConfigurationDescriptions::add, for a service only 1 ParameterSetDescription may be added\n");
      }
    }

    // To minimize the number of copies involved create an empty description first
    // and push it into the vector.  Then perform the copy.
    std::pair<std::string, ParameterSetDescription> pairWithEmptyDescription;
    descriptions_.push_back(pairWithEmptyDescription);
    std::pair<std::string, ParameterSetDescription> & pair = descriptions_.back();

    pair.first = label;
    pair.second = psetDescription;

    if (baseType_ == std::string("ESSource") || baseType_ == std::string("ESProducer")) {
      std::string name("appendToDataLabel");
      if (pair.second.isLabelUnused(name)) {
        pair.second.add<std::string>(name, std::string(""));
      }
    }
  }

  void
  ConfigurationDescriptions::addDefault(ParameterSetDescription const& psetDescription) {

    if (baseType_ == std::string("Source") || baseType_ == std::string("Service")) {
      if (descriptions_.size() != 0U ||
          defaultDescDefined_ == true) {
        throw edm::Exception(edm::errors::LogicError,
          "ConfigurationDescriptions::addDefault, for a source or service only 1 ParameterSetDescription may be added\n");
      }
    }

    defaultDescDefined_ = true;
    defaultDesc_ = psetDescription;

    if (baseType_ == std::string("ESSource") || baseType_ == std::string("ESProducer")) {
      std::string name("appendToDataLabel");
      if (defaultDesc_.isLabelUnused(name)) {
        defaultDesc_.add<std::string>(name, std::string(""));
      }
    }
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
    else if (defaultDescDefined_) {
      defaultDesc_.validate(pset);
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
    if (baseType == std::string("Service") && labelAndDesc.first != pluginName) {
      throw edm::Exception(edm::errors::LogicError,
        "ConfigurationDescriptions::writeCfiForLabel\nFor a service the label and the plugin name must be the same.\n")
        << "This error probably is caused by an incorrect label being passed\nto the ConfigurationDescriptions::add function earlier.\n"
        << "plugin name = \"" << pluginName << "\"  label name = \"" << labelAndDesc.first << "\"\n";
    }

    std::string cfi_filename;
    if (baseType == std::string("Source")) {
      cfi_filename = pluginName + "_cfi.py";
    }
    else {
      cfi_filename = labelAndDesc.first + "_cfi.py";
    }
    std::ofstream outFile(cfi_filename.c_str());


    outFile << "import FWCore.ParameterSet.Config as cms\n\n";
    outFile << labelAndDesc.first << " = cms." << baseType << "('" << pluginName << "'";

    bool startWithComma = true;
    int indentation = 2;
    labelAndDesc.second.writeCfi(outFile, startWithComma, indentation);

    outFile << ")\n";
  
    outFile.close();

    if (baseType == std::string("Source")) {
      std::cout << pluginName << "\n";
    }
    else {
      std::cout << labelAndDesc.first << "\n";
    }
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

    if (descriptions_.empty() && !defaultDescDefined_) {
      char oldFill = os.fill();
      indentation += DocFormatHelper::offsetModuleLabel();
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "There are no PSet descriptions defined for this plugin.\n";
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "PSets will not be validated and no cfi files will be generated.\n";
      os << std::setfill(oldFill);
      if (!brief) os << "\n";
      return;
    }

    if (descriptions_.empty() && defaultDescDefined_ && defaultDesc_.isUnknown()) {
      indentation += DocFormatHelper::offsetModuleLabel();
      char oldFill = os.fill();
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "This plugin has not implemented the function which defines its\n";
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "configuration descriptions yet. No descriptions are available.\n";
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "Its PSets will not be validated, and no cfi files will be generated.\n";
      os << std::setfill(oldFill);
      if (!brief) os << "\n";
      return;
    }

    if (!brief) {
      std::stringstream ss;
      if (defaultDescDefined_) {
        if (descriptions_.empty()) {
          ss << "This plugin has only one PSet description. "
             << "This description is always used to validate configurations. "
             << "Because this configuration has no label, no cfi files will be generated.";
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
             << "The label below is used when generating the cfi file.";
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

    if (defaultDescDefined_) {
      printForLabel(os,
                    std::string("@default"),
                    defaultDesc_,
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

    char oldFill = os.fill();
    os << std::setfill(' ') << std::setw(indentation) << "" << std::setfill(oldFill);
    os << section << " ";
    if (label == std::string("@default")) {
      os << "description without a module label\n";
    }
    else {
      if (!brief) {
        if (baseType_ == std::string("Source") || baseType_ == std::string("Service")) {
          os << "label: ";
        }
        else {
          os << "module label: ";
        }
      }
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
