#ifndef FWCore_ParameterSet_ConfigurationDescriptions_h
#define FWCore_ParameterSet_ConfigurationDescriptions_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ConfigurationDescriptions
//
/**\class ConfigurationDescriptions ConfigurationDescriptions.h FWCore/ParameterSet/interface/ConfigurationDescriptions.h

 Used to hold ParameterSetDescriptions corresponding to labels

*/
//
// Original Author:  W. David Dagenhart
//         Created:  17 December 2008
//

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>
#include <set>
#include <string>
#include <utility>
#include <iosfwd>

namespace edm {

  class ConfigurationDescriptions {
  public:
    typedef std::vector<std::pair<std::string, ParameterSetDescription> >::iterator iterator;

    //NOTE: This does not take ownership of the PreValidatorBase instance so
    // this instance must remain valid for as long as the ConfigurationDescriptions
    // is being modified
    ConfigurationDescriptions(std::string const& baseType, std::string const& pluginName);

    ~ConfigurationDescriptions();

    // ---------------------------------------------------------
    // These functions are used by module developers to create
    // a description for a module.
    // ---------------------------------------------------------

    std::string const& comment() const { return comment_; }
    void setComment(std::string const& value);
    void setComment(char const* value);

    void add(std::string const& label, ParameterSetDescription const& psetDescription);
    void add(char const* label, ParameterSetDescription const& psetDescription);
    void addWithDefaultLabel(ParameterSetDescription const& psetDescription);

    void addDefault(ParameterSetDescription const& psetDescription);

    ///Returns 0 if no default has been assigned
    ParameterSetDescription* defaultDescription();
    iterator begin();
    iterator end();

    // ---------------------------------------------------------
    // These functions use the information in the descriptions
    // ---------------------------------------------------------

    void validate(ParameterSet& pset, std::string const& moduleLabel) const;

    void writeCfis(std::set<std::string>& usedCfiFileNames) const;

    void print(std::ostream& os,
               std::string const& moduleLabel,
               bool brief,
               bool printOnlyLabels,
               size_t lineWidth,
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
                                 std::string const& pluginName,
                                 std::set<std::string>& usedCfiFileNames);

    void printForLabel(std::pair<std::string, ParameterSetDescription> const& labelAndDesc,
                       std::ostream& os,
                       std::string const& moduleLabel,
                       bool brief,
                       bool printOnlyLabels,
                       size_t lineWidth,
                       int indentationn,
                       DescriptionCounter& counter) const;

    void printForLabel(std::ostream& os,
                       std::string const& label,
                       ParameterSetDescription const& description,
                       std::string const& moduleLabel,
                       bool brief,
                       bool printOnlyLabels,
                       size_t lineWidth,
                       int indentationn,
                       DescriptionCounter& counter) const;

    std::string baseType_;
    std::string pluginName_;

    std::vector<std::pair<std::string, ParameterSetDescription> > descriptions_;

    ParameterSetDescription defaultDesc_;

    std::string comment_;
    bool defaultDescDefined_;
  };
}  // namespace edm

#endif
