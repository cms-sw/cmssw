#ifndef FWCore_ParameterSet_ConfigurationDescriptions_h
#define FWCore_ParameterSet_ConfigurationDescriptions_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ConfigurationDescriptions
// 
/**\class ConfigurationDescriptions ConfigurationDescriptions.h FWCore/ParameterSet/interface/ConfigurationDescriptions.h

 Used to hold ParameterSetDescriptions with corresponding module labels

*/
//
// Original Author:  W. David Dagenhart
//         Created:  17 December 2008
//

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>
#include <string>
#include <utility>
#include <iosfwd>

namespace edm {

  class ConfigurationDescriptions {
  public:

    ConfigurationDescriptions();

    // ---------------------------------------------------------
    // These functions are used by module developers to create
    // a description for a module.
    // ---------------------------------------------------------

    std::string const& comment() const { return comment_; }
    void setComment(std::string const & value);
    void setComment(char const* value);

    void add(std::string const& label, ParameterSetDescription const& psetDescription);
    void add(char const* label, ParameterSetDescription const& psetDescription);

    void addUnknownLabel(ParameterSetDescription const& psetDescription);

    // ---------------------------------------------------------
    // These functions use the information in the descriptions
    // ---------------------------------------------------------

    void validate(ParameterSet & pset, std::string const& moduleLabel) const;

    void writeCfis(std::string const& baseType,
                   std::string const& pluginName) const;

    void print(std::ostream & os,
               std::string const& moduleLabel,
               bool brief,
               bool printOnlyLabels,
               unsigned lineWidth,
               int indentation,
               int iPlugin) const;

    // ---------------------------------------------------------

  private:

    class DescriptionCounter {
    public:
      int iPlugin;
      int iSelectedModule;
      int iModule;
    };

    static void writeCfiForLabel(std::pair<std::string, ParameterSetDescription> const& labelAndDesc,
                                 std::string const& baseType,
                                 std::string const& pluginName);

    void printForLabel(std::pair<std::string, ParameterSetDescription> const& labelAndDesc,
                       std::ostream & os,
                       std::string const& moduleLabel,
                       bool brief,
                       bool printOnlyLabels,
                       unsigned lineWidth,
                       int indentationn,
                       DescriptionCounter & counter) const;

    void printForLabel(std::ostream & os,
                       std::string const& label,
                       ParameterSetDescription const& description,
                       std::string const& moduleLabel,
                       bool brief,
                       bool printOnlyLabels,
                       unsigned lineWidth,
                       int indentationn,
                       DescriptionCounter & counter) const;

    std::vector<std::pair<std::string, ParameterSetDescription> > descriptions_;

    bool unknownDescDefined_;
    ParameterSetDescription descForUnknownLabels_;

    std::string comment_;
  };
}

#endif
