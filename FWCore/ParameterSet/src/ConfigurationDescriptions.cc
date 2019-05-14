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
#include "FWCore/ParameterSet/interface/defaultModuleLabel.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <cstring>

namespace {
  void matchLabel(std::pair<std::string, edm::ParameterSetDescription> const& thePair,
                  std::string const& moduleLabel,
                  edm::ParameterSetDescription const*& psetDesc) {
    if (thePair.first == moduleLabel) {
      psetDesc = &thePair.second;
    }
  }
}  // namespace

static const char* const kSource = "Source";
static const char* const kService = "Service";
static const char* const k_source = "source";

namespace edm {

  ConfigurationDescriptions::ConfigurationDescriptions(std::string const& baseType, std::string const& pluginName)
      : baseType_(baseType), pluginName_(pluginName), defaultDescDefined_(false) {}

  ConfigurationDescriptions::~ConfigurationDescriptions() {}

  void ConfigurationDescriptions::setComment(std::string const& value) { comment_ = value; }

  void ConfigurationDescriptions::setComment(char const* value) { comment_ = value; }

  void ConfigurationDescriptions::add(char const* label, ParameterSetDescription const& psetDescription) {
    std::string labelString(label);
    add(labelString, psetDescription);
  }

  void ConfigurationDescriptions::add(std::string const& label, ParameterSetDescription const& psetDescription) {
    if (0 == strcmp(baseType_.c_str(), kSource)) {
      if (0 != strcmp(label.c_str(), k_source)) {
        throw edm::Exception(edm::errors::LogicError,
                             "ConfigurationDescriptions::add, when adding a ParameterSetDescription for a source the "
                             "label must be \"source\"\n");
      }
      if (!descriptions_.empty() || defaultDescDefined_ == true) {
        throw edm::Exception(
            edm::errors::LogicError,
            "ConfigurationDescriptions::add, for a source only 1 ParameterSetDescription may be added\n");
      }
    } else if (0 == strcmp(baseType_.c_str(), kService)) {
      if (!descriptions_.empty() || defaultDescDefined_ == true) {
        throw edm::Exception(
            edm::errors::LogicError,
            "ConfigurationDescriptions::add, for a service only 1 ParameterSetDescription may be added\n");
      }
    }

    // To minimize the number of copies involved create an empty description first
    // and push it into the vector.  Then perform the copy.
    std::pair<std::string, ParameterSetDescription> pairWithEmptyDescription;
    descriptions_.push_back(pairWithEmptyDescription);
    std::pair<std::string, ParameterSetDescription>& pair = descriptions_.back();

    pair.first = label;
    pair.second = psetDescription;
  }

  void ConfigurationDescriptions::addWithDefaultLabel(ParameterSetDescription const& psetDescription) {
    std::string label;
    if (kService == baseType_) {
      label = pluginName_;
    } else if (kSource == baseType_) {
      label = "source";
    } else {
      label = defaultModuleLabel(pluginName_);
    }
    add(label, psetDescription);
  }

  void ConfigurationDescriptions::addDefault(ParameterSetDescription const& psetDescription) {
    if (0 == strcmp(baseType_.c_str(), kSource) || 0 == strcmp(baseType_.c_str(), kService)) {
      if (!descriptions_.empty() || defaultDescDefined_ == true) {
        throw edm::Exception(edm::errors::LogicError,
                             "ConfigurationDescriptions::addDefault, for a source or service only 1 "
                             "ParameterSetDescription may be added\n");
      }
    }

    defaultDescDefined_ = true;
    defaultDesc_ = psetDescription;
  }

  ParameterSetDescription* ConfigurationDescriptions::defaultDescription() {
    if (defaultDescDefined_) {
      return &defaultDesc_;
    }
    return nullptr;
  }

  ConfigurationDescriptions::iterator ConfigurationDescriptions::begin() { return descriptions_.begin(); }

  ConfigurationDescriptions::iterator ConfigurationDescriptions::end() { return descriptions_.end(); }

  void ConfigurationDescriptions::validate(ParameterSet& pset, std::string const& moduleLabel) const {
    ParameterSetDescription const* psetDesc = nullptr;
    for_all(descriptions_, std::bind(&matchLabel, std::placeholders::_1, std::cref(moduleLabel), std::ref(psetDesc)));

    // If there is a matching label
    if (psetDesc != nullptr) {
      psetDesc->validate(pset);
    }
    // Is there an explicit description to be used for a non standard label
    else if (defaultDescDefined_) {
      defaultDesc_.validate(pset);
    }
    // Otherwise use the first one.
    else if (!descriptions_.empty()) {
      descriptions_[0].second.validate(pset);
    }
    // It is possible for no descriptions to be defined and no validation occurs
    // for this module ever.
  }

  void ConfigurationDescriptions::writeCfis(std::set<std::string>& usedCfiFileNames) const {
    for_all(descriptions_,
            std::bind(&ConfigurationDescriptions::writeCfiForLabel,
                      std::placeholders::_1,
                      std::cref(baseType_),
                      std::cref(pluginName_),
                      std::ref(usedCfiFileNames)));
  }

  void ConfigurationDescriptions::writeCfiForLabel(std::pair<std::string, ParameterSetDescription> const& labelAndDesc,
                                                   std::string const& baseType,
                                                   std::string const& pluginName,
                                                   std::set<std::string>& usedCfiFileNames) {
    if (0 == strcmp(baseType.c_str(), kService) && labelAndDesc.first != pluginName) {
      throw edm::Exception(edm::errors::LogicError,
                           "ConfigurationDescriptions::writeCfiForLabel\nFor a service the label and the plugin name "
                           "must be the same.\n")
          << "This error is probably caused by an incorrect label being passed\nto the ConfigurationDescriptions::add "
             "function earlier.\n"
          << "plugin name = \"" << pluginName << "\"  label name = \"" << labelAndDesc.first << "\"\n";
    }

    std::string cfi_filename;
    if (0 == strcmp(baseType.c_str(), kSource)) {
      cfi_filename = pluginName + "_cfi.py";
    } else {
      cfi_filename = labelAndDesc.first + "_cfi.py";
    }
    if (!usedCfiFileNames.insert(cfi_filename).second) {
      edm::Exception ex(edm::errors::LogicError,
                        "Two cfi files are being generated with the same name in the same directory.\n");
      ex << "The cfi file name is '" << cfi_filename << "' and\n"
         << "the module label is \'" << labelAndDesc.first << "\'.\n"
         << "This error is probably caused by an error in one or more fillDescriptions functions\n"
         << "where duplicate module labels are being passed to the ConfigurationDescriptions::add\n"
         << "function. All such module labels must be unique within a package.\n"
         << "If you do not want the generated cfi file and do not need more than one\n"
         << "description for a plugin, then a way to fix this is to use the addDefault\n"
         << "function instead of the add function.\n"
         << "There are 3 common ways this problem can happen.\n"
         << "1. This can happen when a module label is explicitly duplicated in one or more\n"
         << "fillDescriptions functions. Fix these by changing the module labels to be unique.\n"
         << "2. This can also happen when a module class is a template class and plugins are\n"
         << "defined by instantiations with differing template parameters and these plugins\n"
         << "share the same fillDescriptions function. Fix these by specializing the fillDescriptions\n"
         << "function for each template instantiation.\n"
         << "3. This can also happen when there is an inheritance heirarchy and multiple plugin modules\n"
         << "are defined using derived classes and the base class which share the same fillDescriptions\n"
         << "function. Fix these by redefining the fillDescriptions function in each derived class.\n";
      ex.addContext("Executing function ConfigurationDescriptions::writeCfiForLabel");
      throw ex;
    }
    std::ofstream outFile(cfi_filename.c_str());
    if (outFile.fail()) {
      edm::Exception ex(edm::errors::LogicError, "Creating cfi file failed.\n");
      ex << "Opening a file '" << cfi_filename << "' for module '" << labelAndDesc.first << "' failed.\n";
      ex << "Error code from errno " << errno << ": " << std::strerror(errno) << "\n";

      ex.addContext("Executing function ConfigurationDescriptions::writeCfiForLabel");
      throw ex;
    }

    outFile << "import FWCore.ParameterSet.Config as cms\n\n";
    outFile << labelAndDesc.first << " = cms." << baseType << "('" << pluginName << "'";

    bool startWithComma = true;
    int indentation = 2;
    labelAndDesc.second.writeCfi(outFile, startWithComma, indentation);

    outFile << ")\n";

    outFile.close();

    if (0 == strcmp(baseType.c_str(), kSource)) {
      std::cout << pluginName << "\n";
    } else {
      std::cout << labelAndDesc.first << "\n";
    }
  }

  void ConfigurationDescriptions::print(std::ostream& os,
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
      if (!brief)
        os << "\n";
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
      if (!brief)
        os << "\n";
      return;
    }

    if (!brief) {
      std::stringstream ss;
      if (defaultDescDefined_) {
        if (descriptions_.empty()) {
          ss << "This plugin has only one PSet description. "
             << "This description is always used to validate configurations. "
             << "Because this configuration has no label, no cfi files will be generated.";
        } else {
          ss << "This plugin has " << (descriptions_.size() + 1U) << " PSet descriptions. "
             << "The description used to validate a configuration is selected by "
             << "matching the module labels. If none match, then the last description, "
             << "which has no label, is selected. "
             << "A cfi file will be generated for each configuration with a module label.";
        }
      } else {
        if (descriptions_.size() == 1U) {
          ss << "This plugin has " << descriptions_.size() << " PSet description. "
             << "This description is always used to validate configurations. "
             << "The label below is used when generating the cfi file.";
        } else {
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

    for (auto const& d : descriptions_) {
      printForLabel(d, os, moduleLabel, brief, printOnlyLabels, lineWidth, indentation, counter);
    }

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

  void ConfigurationDescriptions::printForLabel(std::pair<std::string, ParameterSetDescription> const& labelAndDesc,
                                                std::ostream& os,
                                                std::string const& moduleLabel,
                                                bool brief,
                                                bool printOnlyLabels,
                                                size_t lineWidth,
                                                int indentation,
                                                DescriptionCounter& counter) const {
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

  void ConfigurationDescriptions::printForLabel(std::ostream& os,
                                                std::string const& label,
                                                ParameterSetDescription const& description,
                                                std::string const& moduleLabel,
                                                bool brief,
                                                bool printOnlyLabels,
                                                size_t lineWidth,
                                                int indentation,
                                                DescriptionCounter& counter) const {
    ++counter.iModule;
    if (!moduleLabel.empty() && label != moduleLabel)
      return;
    ++counter.iSelectedModule;

    std::stringstream ss;
    ss << counter.iPlugin << "." << counter.iSelectedModule;
    std::string section = ss.str();

    char oldFill = os.fill();
    os << std::setfill(' ') << std::setw(indentation) << "" << std::setfill(oldFill);
    os << section << " ";
    if (label == std::string("@default")) {
      os << "description without a module label\n";
    } else {
      if (!brief) {
        if (0 == strcmp(baseType_.c_str(), kSource) || 0 == strcmp(baseType_.c_str(), kService)) {
          os << "label: ";
        } else {
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
    if (printOnlyLabels)
      return;

    DocFormatHelper dfh;
    dfh.setBrief(brief);
    dfh.setLineWidth(lineWidth);
    dfh.setIndentation(indentation + DocFormatHelper::offsetTopLevelPSet());
    dfh.setSection(section);
    dfh.setParent(DocFormatHelper::TOP);

    description.print(os, dfh);
  }
}  // namespace edm
