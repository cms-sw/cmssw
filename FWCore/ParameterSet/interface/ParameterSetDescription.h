#ifndef FWCore_ParameterSet_ParameterSetDescription_h
#define FWCore_ParameterSet_ParameterSetDescription_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterSetDescription
//
/**\class ParameterSetDescription ParameterSetDescription.h FWCore/ParameterSet/interface/ParameterSetDescription.h

 Description: Used to describe the allowed values in a ParameterSet

 Usage:
    <usage>


 Implementation Details:

    Note that there are some comments in the file ParameterDescriptionNode.h
    that might be useful for someone attempting to understand the implementation
    details.  This class holds a container full of nodes.  One node can represent
    a description of a single parameter or some logical restriction on the
    combinations of parameters allowed in a ParameterSet.  Often these logical
    restrictions are implemented by the nodes themselves containing a tree
    structure of other nodes.
*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 31 15:18:40 EDT 2007
//

#include "FWCore/Utilities/interface/value_ptr.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"

#include <vector>
#include <set>
#include <string>
#include <memory>
#include <iosfwd>

namespace edm {

  class ParameterSet;
  class ParameterDescriptionBase;
  class ParameterWildcardBase;
  class ParameterDescriptionNode;
  template <typename T>
  class ParameterDescription;
  template <typename T>
  class ParameterDescriptionCases;
  class DocFormatHelper;

  class ParameterSetDescription {
  public:
    class SetDescriptionEntry {
    public:
      bool optional() const { return optional_; }
      bool writeToCfi() const { return writeToCfi_; }
      edm::value_ptr<ParameterDescriptionNode> const& node() const { return node_; }
      void setOptional(bool value) { optional_ = value; }
      void setWriteToCfi(bool value) { writeToCfi_ = value; }
      ParameterDescriptionNode* setNode(std::unique_ptr<ParameterDescriptionNode> node) {
        node_ = std::move(node);
        return node_.operator->();
      }

    private:
      bool optional_;
      bool writeToCfi_;
      edm::value_ptr<ParameterDescriptionNode> node_;
    };

    typedef std::vector<SetDescriptionEntry> SetDescriptionEntries;
    typedef SetDescriptionEntries::const_iterator const_iterator;

    ParameterSetDescription();
    virtual ~ParameterSetDescription();

    std::string const& comment() const { return comment_; }
    void setComment(std::string const& value);
    void setComment(char const* value);

    ///allow any parameter label/value pairs
    void setAllowAnything();

    // This is set only for parameterizables which have not set their descriptions.
    // This should only be called to allow backwards compatibility.
    void setUnknown();

    // ***** In these next 8 functions named "add", T is the parameter type ******
    // Exceptions: For parameters of type ParameterSet, T should be a
    // ParameterSetDescription instead of a ParameterSet.  And do not
    // use these next 8 functions for parameters of type vector<ParameterSet>

    template <typename T, typename U>
    ParameterDescriptionBase* add(U const& iLabel, T const& value) {
      return add<T, U>(iLabel, value, true, false, true);
    }

    template <typename T, typename U>
    ParameterDescriptionBase* addUntracked(U const& iLabel, T const& value) {
      return add<T, U>(iLabel, value, false, false, true);
    }

    template <typename T, typename U>
    ParameterDescriptionBase* addOptional(U const& iLabel, T const& value) {
      return add<T, U>(iLabel, value, true, true, true);
    }

    template <typename T, typename U>
    ParameterDescriptionBase* addOptionalUntracked(U const& iLabel, T const& value) {
      return add<T, U>(iLabel, value, false, true, true);
    }

    // For the next 4 functions, there is no default so they will not get injected
    // during validation if missing and they will not get written into cfi files.

    template <typename T, typename U>
    ParameterDescriptionBase* add(U const& iLabel) {
      return add<T, U>(iLabel, true, false, true);
    }

    template <typename T, typename U>
    ParameterDescriptionBase* addUntracked(U const& iLabel) {
      return add<T, U>(iLabel, false, false, true);
    }

    template <typename T, typename U>
    ParameterDescriptionBase* addOptional(U const& iLabel) {
      return add<T, U>(iLabel, true, true, true);
    }

    template <typename T, typename U>
    ParameterDescriptionBase* addOptionalUntracked(U const& iLabel) {
      return add<T, U>(iLabel, false, true, true);
    }

    // ***** Use these 8 functions for parameters of type vector<ParameterSet> *****
    // When a vector<ParameterSet> appears in a configuration, all of its
    // elements will be validated using the description in the argument named
    // "validator" below.  The argument named "defaults" is used when the
    // a vector<ParameterSet> is required to be in the configuration and
    // is absent.  Note that these default ParameterSet's will be validated
    // as if they had appeared in the configuration so they must be consistent
    // with the description and missing parameters that have defaults in
    // in the description will be inserted during validation.  These defaults
    // are also used when writing cfi files.

    template <typename U>
    ParameterDescriptionBase* addVPSet(U const& iLabel,
                                       ParameterSetDescription const& validator,
                                       std::vector<ParameterSet> const& defaults) {
      return addVPSet<U>(iLabel, validator, defaults, true, false, true);
    }

    template <typename U>
    ParameterDescriptionBase* addVPSetUntracked(U const& iLabel,
                                                ParameterSetDescription const& validator,
                                                std::vector<ParameterSet> const& defaults) {
      return addVPSet<U>(iLabel, validator, defaults, false, false, true);
    }

    template <typename U>
    ParameterDescriptionBase* addVPSetOptional(U const& iLabel,
                                               ParameterSetDescription const& validator,
                                               std::vector<ParameterSet> const& defaults) {
      return addVPSet<U>(iLabel, validator, defaults, true, true, true);
    }

    template <typename U>
    ParameterDescriptionBase* addVPSetOptionalUntracked(U const& iLabel,
                                                        ParameterSetDescription const& validator,
                                                        std::vector<ParameterSet> const& defaults) {
      return addVPSet<U>(iLabel, validator, defaults, false, true, true);
    }

    template <typename U>
    ParameterDescriptionBase* addVPSet(U const& iLabel, ParameterSetDescription const& validator) {
      return addVPSet<U>(iLabel, validator, true, false, true);
    }

    template <typename U>
    ParameterDescriptionBase* addVPSetUntracked(U const& iLabel, ParameterSetDescription const& validator) {
      return addVPSet<U>(iLabel, validator, false, false, true);
    }

    template <typename U>
    ParameterDescriptionBase* addVPSetOptional(U const& iLabel, ParameterSetDescription const& validator) {
      return addVPSet<U>(iLabel, validator, true, true, true);
    }

    template <typename U>
    ParameterDescriptionBase* addVPSetOptionalUntracked(U const& iLabel, ParameterSetDescription const& validator) {
      return addVPSet<U>(iLabel, validator, false, true, true);
    }

    // ********* Wildcards *********

    template <typename T, typename U>
    ParameterWildcardBase* addWildcard(U const& pattern) {
      return addWildcard<T, U>(pattern, true);
    }

    template <typename T, typename U>
    ParameterWildcardBase* addWildcardUntracked(U const& pattern) {
      return addWildcard<T, U>(pattern, false);
    }

    // ********* Used to insert generic nodes of any type ************

    ParameterDescriptionNode* addNode(ParameterDescriptionNode const& node);
    ParameterDescriptionNode* addNode(std::unique_ptr<ParameterDescriptionNode> node);
    ParameterDescriptionNode* addOptionalNode(ParameterDescriptionNode const& node, bool writeToCfi);
    ParameterDescriptionNode* addOptionalNode(std::unique_ptr<ParameterDescriptionNode> node, bool writeToCfi);

    // ********* Switches ************
    // ifValue will only work with type T as a bool, int, or string.
    // T holds the value of the switch variable.
    // If you try using any other type, then it will not compile.
    template <typename T>
    ParameterDescriptionNode* ifValue(ParameterDescription<T> const& switchParameter,
                                      std::unique_ptr<ParameterDescriptionCases<T>> cases) {
      return ifValue<T>(switchParameter, std::move(cases), false, true);
    }

    template <typename T>
    ParameterDescriptionNode* ifValueOptional(ParameterDescription<T> const& switchParameter,
                                              std::unique_ptr<ParameterDescriptionCases<T>> cases,
                                              bool writeToCfi) {
      return ifValue<T>(switchParameter, std::move(cases), true, writeToCfi);
    }

    // ********* if exists ************
    ParameterDescriptionNode* ifExists(ParameterDescriptionNode const& node1, ParameterDescriptionNode const& node2) {
      return ifExists(node1, node2, false, true);
    }

    ParameterDescriptionNode* ifExistsOptional(ParameterDescriptionNode const& node1,
                                               ParameterDescriptionNode const& node2,
                                               bool writeToCfi) {
      return ifExists(node1, node2, true, writeToCfi);
    }

    // ********* for parameters that are a list of allowed labels *********
    template <typename T, typename U>
    ParameterDescriptionNode* labelsFrom(U const& iLabel) {
      return labelsFrom<T, U>(iLabel, true, false, true);
    }

    template <typename T, typename U>
    ParameterDescriptionNode* labelsFromUntracked(U const& iLabel) {
      return labelsFrom<T, U>(iLabel, false, false, true);
    }

    template <typename T, typename U>
    ParameterDescriptionNode* labelsFromOptional(U const& iLabel, bool writeToCfi) {
      return labelsFrom<T, U>(iLabel, true, true, writeToCfi);
    }

    template <typename T, typename U>
    ParameterDescriptionNode* labelsFromOptionalUntracked(U const& iLabel, bool writeToCfi) {
      return labelsFrom<T, U>(iLabel, false, true, writeToCfi);
    }

    // These next four functions only work when the template
    // parameters are:
    // T = ParameterSetDescription and V = ParameterSetDescription
    // or
    // T = vector<ParameterSet> and V = ParameterSetDescription
    // In either case U can be either a string or char*
    // Note the U and V can be determined from the arguments, but
    // T must be explicitly specified by the calling function.
    template <typename T, typename U, typename V>
    ParameterDescriptionNode* labelsFrom(U const& iLabel, V const& desc) {
      return labelsFrom<T, U, V>(iLabel, true, false, true, desc);
    }

    template <typename T, typename U, typename V>
    ParameterDescriptionNode* labelsFromUntracked(U const& iLabel, V const& desc) {
      return labelsFrom<T, U, V>(iLabel, false, false, true, desc);
    }

    template <typename T, typename U, typename V>
    ParameterDescriptionNode* labelsFromOptional(U const& iLabel, bool writeToCfi, V const& desc) {
      return labelsFrom<T, U, V>(iLabel, true, true, writeToCfi, desc);
    }

    template <typename T, typename U, typename V>
    ParameterDescriptionNode* labelsFromOptionalUntracked(U const& iLabel, bool writeToCfi, V const& desc) {
      return labelsFrom<T, U, V>(iLabel, false, true, writeToCfi, desc);
    }

    bool anythingAllowed() const { return anythingAllowed_; }
    bool isUnknown() const { return unknown_; }

    const_iterator begin() const { return entries_.begin(); }

    const_iterator end() const { return entries_.end(); }

    // Better performance if space is reserved for the number of
    // top level parameters before any are added.
    void reserve(SetDescriptionEntries::size_type n) { entries_.reserve(n); }

    void validate(ParameterSet& pset) const;

    void writeCfi(std::ostream& os, bool startWithComma, int indentation) const;

    void print(std::ostream& os, DocFormatHelper& dfh) const;

    bool isLabelUnused(std::string const& label) const;

  private:
    template <typename T, typename U>
    ParameterDescriptionBase* add(U const& iLabel, T const& value, bool isTracked, bool isOptional, bool writeToCfi);

    template <typename T, typename U>
    ParameterDescriptionBase* add(U const& iLabel, bool isTracked, bool isOptional, bool writeToCfi);

    template <typename U>
    ParameterDescriptionBase* addVPSet(U const& iLabel,
                                       ParameterSetDescription const& validator,
                                       std::vector<ParameterSet> const& defaults,
                                       bool isTracked,
                                       bool isOptional,
                                       bool writeToCfi);

    template <typename U>
    ParameterDescriptionBase* addVPSet(
        U const& iLabel, ParameterSetDescription const& validator, bool isTracked, bool isOptional, bool writeToCfi);

    template <typename T, typename U>
    ParameterWildcardBase* addWildcard(U const& pattern, bool isTracked);

    ParameterDescriptionNode* addNode(std::unique_ptr<ParameterDescriptionNode> node, bool optional, bool writeToCfi);

    template <typename T>
    ParameterDescriptionNode* ifValue(ParameterDescription<T> const& switchParameter,
                                      std::unique_ptr<ParameterDescriptionCases<T>> cases,
                                      bool optional,
                                      bool writeToCfi);

    ParameterDescriptionNode* ifExists(ParameterDescriptionNode const& node1,
                                       ParameterDescriptionNode const& node2,
                                       bool optional,
                                       bool writeToCfi);

    template <typename T, typename U>
    ParameterDescriptionNode* labelsFrom(U const& iLabel, bool isTracked, bool optional, bool writeToCfi);

    template <typename T, typename U, typename V>
    ParameterDescriptionNode* labelsFrom(U const& iLabel, bool isTracked, bool optional, bool writeToCfi, V const& desc);

    static void validateNode(SetDescriptionEntry const& entry,
                             ParameterSet& pset,
                             std::set<std::string>& validatedNames);

    static void throwIllegalParameters(std::vector<std::string> const& parameterNames,
                                       std::set<std::string> const& validatedNames);

    static void writeNode(SetDescriptionEntry const& entry,
                          std::ostream& os,
                          bool& startWithComma,
                          int indentation,
                          bool& wroteSomething);

    static void printNode(SetDescriptionEntry const& entry, std::ostream& os, DocFormatHelper& dfh);

    void throwIfLabelsAlreadyUsed(std::set<std::string> const& nodeLabels);
    void throwIfWildcardCollision(std::set<ParameterTypes> const& nodeParameterTypes,
                                  std::set<ParameterTypes> const& nodeWildcardTypes);

    bool anythingAllowed_;
    bool unknown_;
    SetDescriptionEntries entries_;

    std::set<std::string> usedLabels_;
    std::set<ParameterTypes> typesUsedForParameters_;
    std::set<ParameterTypes> typesUsedForWildcards_;

    std::string comment_;
  };
}  // namespace edm

#include "FWCore/ParameterSet/interface/ParameterWildcard.h"
#include "FWCore/ParameterSet/interface/ParameterSwitch.h"
#include "FWCore/ParameterSet/interface/AllowedLabelsDescription.h"

namespace edm {

  template <typename T, typename U>
  ParameterDescriptionBase* ParameterSetDescription::add(
      U const& iLabel, T const& value, bool isTracked, bool isOptional, bool writeToCfi) {
    std::unique_ptr<ParameterDescriptionNode> node =
        std::make_unique<ParameterDescription<T>>(iLabel, value, isTracked);
    ParameterDescriptionNode* pnode = addNode(std::move(node), isOptional, writeToCfi);
    return static_cast<ParameterDescriptionBase*>(pnode);
  }

  template <typename T, typename U>
  ParameterDescriptionBase* ParameterSetDescription::add(U const& iLabel,
                                                         bool isTracked,
                                                         bool isOptional,
                                                         bool writeToCfi) {
    std::unique_ptr<ParameterDescriptionNode> node = std::make_unique<ParameterDescription<T>>(iLabel, isTracked);
    ParameterDescriptionNode* pnode = addNode(std::move(node), isOptional, writeToCfi);
    return static_cast<ParameterDescriptionBase*>(pnode);
  }

  template <typename U>
  ParameterDescriptionBase* ParameterSetDescription::addVPSet(U const& iLabel,
                                                              ParameterSetDescription const& validator,
                                                              std::vector<ParameterSet> const& defaults,
                                                              bool isTracked,
                                                              bool isOptional,
                                                              bool writeToCfi) {
    std::unique_ptr<ParameterDescriptionNode> node =
        std::make_unique<ParameterDescription<std::vector<ParameterSet>>>(iLabel, validator, isTracked, defaults);
    ParameterDescriptionNode* pnode = addNode(std::move(node), isOptional, writeToCfi);
    return static_cast<ParameterDescriptionBase*>(pnode);
  }

  template <typename U>
  ParameterDescriptionBase* ParameterSetDescription::addVPSet(
      U const& iLabel, ParameterSetDescription const& validator, bool isTracked, bool isOptional, bool writeToCfi) {
    std::unique_ptr<ParameterDescriptionNode> node =
        std::make_unique<ParameterDescription<std::vector<ParameterSet>>>(iLabel, validator, isTracked);
    ParameterDescriptionNode* pnode = addNode(std::move(node), isOptional, writeToCfi);
    return static_cast<ParameterDescriptionBase*>(pnode);
  }

  template <typename T, typename U>
  ParameterWildcardBase* ParameterSetDescription::addWildcard(U const& pattern, bool isTracked) {
    std::unique_ptr<ParameterDescriptionNode> node =
        std::make_unique<ParameterWildcard<T>>(pattern, RequireZeroOrMore, isTracked);
    ParameterDescriptionNode* pnode = addNode(std::move(node), true, false);
    return static_cast<ParameterWildcardBase*>(pnode);
  }

  template <typename T>
  ParameterDescriptionNode* ParameterSetDescription::ifValue(ParameterDescription<T> const& switchParameter,
                                                             std::unique_ptr<ParameterDescriptionCases<T>> cases,
                                                             bool optional,
                                                             bool writeToCfi) {
    std::unique_ptr<ParameterDescriptionNode> pdswitch =
        std::make_unique<ParameterSwitch<T>>(switchParameter, std::move(cases));
    return addNode(std::move(pdswitch), optional, writeToCfi);
  }

  template <typename T, typename U>
  ParameterDescriptionNode* ParameterSetDescription::labelsFrom(U const& iLabel,
                                                                bool isTracked,
                                                                bool optional,
                                                                bool writeToCfi) {
    std::unique_ptr<ParameterDescriptionNode> pd = std::make_unique<AllowedLabelsDescription<T>>(iLabel, isTracked);
    return addNode(std::move(pd), optional, writeToCfi);
  }

  template <typename T, typename U, typename V>
  ParameterDescriptionNode* ParameterSetDescription::labelsFrom(
      U const& iLabel, bool isTracked, bool optional, bool writeToCfi, V const& desc) {
    std::unique_ptr<ParameterDescriptionNode> pd =
        std::make_unique<AllowedLabelsDescription<T>>(iLabel, desc, isTracked);
    return addNode(std::move(pd), optional, writeToCfi);
  }
}  // namespace edm

#endif
