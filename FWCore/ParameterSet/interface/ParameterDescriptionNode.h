#ifndef FWCore_ParameterSet_ParameterDescriptionNode_h
#define FWCore_ParameterSet_ParameterDescriptionNode_h

// This is a base class for the class that describes
// the parameters that are allowed or required to be
// in a ParameterSet.  It is also a base class for
// other more complex logical structures which describe
// which combinations of parameters are allowed to be
// in a ParameterSet.

#include "FWCore/Utilities/interface/value_ptr.h"

#include <string>
#include <set>
#include <iosfwd>
#include <memory>

namespace edm {

  class ParameterSet;
  template <typename T> class ParameterDescriptionCases;
  class DocFormatHelper;

  // Originally these were defined such that the values were the
  // same as in the ParameterSet Entry class and the validation
  // depended on that.  But at the moment I'm typing this comment,
  // the code no longer depends on the values being the same (which
  // is probably good because nothing enforces the correspondence,
  // a task for the future when someone has free time would be
  // to define the values in a common header, but that would involve
  // significant changes to ParameterSet ...)
  enum ParameterTypes {
    k_int32 = 'I',
    k_vint32 = 'i',
    k_uint32 = 'U',
    k_vuint32 = 'u',
    k_int64 = 'L',
    k_vint64 = 'l',
    k_uint64 = 'X',
    k_vuint64 = 'x',
    k_double = 'D',
    k_vdouble = 'd',
    k_bool = 'B',
    k_string = 'S',
    k_vstring = 's',
    k_EventID = 'E',
    k_VEventID = 'e',
    k_LuminosityBlockID = 'M',
    k_VLuminosityBlockID = 'm',
    k_InputTag = 't',
    k_VInputTag = 'v',
    k_FileInPath = 'F',
    k_LuminosityBlockRange = 'A',
    k_VLuminosityBlockRange = 'a',
    k_EventRange = 'R',
    k_VEventRange = 'r',
    k_PSet = 'Q',
    k_VPSet = 'q'
  };

  std::string parameterTypeEnumToString(ParameterTypes iType);

  struct ParameterTypeToEnum {
    template <class T>
    static ParameterTypes toEnum();
  };


  class ParameterDescriptionNode {

  public:

    virtual ~ParameterDescriptionNode();

    virtual ParameterDescriptionNode* clone() const = 0;

    std::string const& comment() const { return comment_; }
    void setComment(std::string const& value);
    void setComment(char const* value);

    // The validate function should do one of three things, find that the
    // node "exists", make the node "exist" by inserting missing parameters
    // or throw.  The only exception to this rule occurs when the argument
    // named "optional" is true, which should only be possible for the
    // top level nodes of a ParameterSetDescription.  When a parameter is
    // found or inserted its label is added into the list of validatedLabels.
    void validate(ParameterSet& pset,
                  std::set<std::string>& validatedLabels,
                  bool optional) const {
      validate_(pset, validatedLabels, optional);
    }

    // As long as it has default values, this will attempt to write
    // parameters associated with a node into a cfi file that is
    // being automatically generated.  It is quite possible for
    // to produce a cfi that will fail validation.  In some cases,
    // this will imply the user is required to supply certain missing
    // parameters that do not appear in the cfi and do not have defaults
    // in the description.  It is also possible to create a pathological
    // ParameterSetDescription where the algorithm fails to write
    // a valid cfi, in some cases the description can be so pathological
    // that it is impossible to write a cfi that will pass validation.
    void writeCfi(std::ostream& os,
                  bool& startWithComma,
                  int indentation,
                  bool& wroteSomething) const {
      writeCfi_(os, startWithComma, indentation, wroteSomething);
    }

    // Print out the description in human readable format
    void print(std::ostream& os,
               bool optional,
               bool writeToCfi,
               DocFormatHelper& dfh);

    bool hasNestedContent() {
      return hasNestedContent_();
    }

    void printNestedContent(std::ostream& os,
                            bool optional,
                            DocFormatHelper& dfh);

    // The next three functions are only called by the logical nodes
    // on their subnodes.  When executing these functions, the
    // insertion of missing parameters does not occur.

    // Usually checks to see if a parameter exists in the configuration, but
    // if the node is a logical node, then it returns the value of the logical
    // expression.
    bool exists(ParameterSet const& pset) const {
      return exists_(pset);
    }

    // For most nodes, this simply returns the same value as the exists
    // function.  But for AND nodes this returns true if either its subnodes
    // exists.  Used by operator&& during validation, if either of an AND node's
    // subnodes exists, then both subnodes get validated.
    bool partiallyExists(ParameterSet const& pset) const {
      return partiallyExists_(pset);
    }

    // For most nodes, this simply returns the same value as the exists
    // function. It is different for an XOR node.  It counts
    // XOR subnodes whose exists function returns true.  And it
    // does this recursively into XOR nodes that are contained in
    // other XOR nodes.
    // Used by operator^ during validation:
    // -- if it returns more than 1, then validation will throw,
    // -- if it returns exactly one, then only the nonzero subnode gets validated
    // -- if it returns zero, then validation tries to validate the first node and
    // then rechecks to see what the missing parameter insertion did (there could
    // be side effects on the nodes that were not validated)
    int howManyXORSubNodesExist(ParameterSet const& pset) const {
      return howManyXORSubNodesExist_(pset);
    }

    /* Validation puts requirements on which parameters can and cannot exist
    within a ParameterSet.  The evaluation of whether a ParameterSet passes
    or fails the rules in the ParameterSetDescription is complicated by
    the fact that we allow missing parameters to be injected into the
    ParameterSet during validation.  One must worry whether injecting a
    missing parameter invalidates some other part of the ParameterSet that
    was already checked and determined to be OK.  The following restrictions
    avoid that problem.

        - The same parameter labels cannot occur in different nodes of the
        same ParameterSetDescription.  There are two exceptions to this.
        Nodes that are contained in the cases of a ParameterSwitch or the
        subnodes of an "exclusive or" are allowed to use the same labels.

        - If insertion is necessary to make an "exclusive or" node pass
        validation, then the insertion could make more than one of the
        possibilities evaluate true.  This must be checked for after the
        insertions occur. The behavior is to throw a Configuration exception
        if this problem is encountered. (Example: (A && B) ^ (A && C) where
        C already exists in the ParameterSet but A and B do not.  A and B
        get inserted by the algorithm, because it tries to make the first
        possibility true when all fail without insertion.  Then both
        parts of the "exclusive or" pass, which is a validation failure).

        - Another potential problem is that a parameter insertion related
        to one ParameterDescription could match unrelated wildcards causing
        other validation requirements to change from being passing to failing
        or vice versa.  This makes it almost impossible to determine if a
        ParameterSet passes validation.  Each time you try to loop through
        and check, the result of validation could change.  To avoid this problem,
        a list is maintained of the type for all wildcards.  Another list is
        maintained for the type of all parameters.  As new items are added
        we check for collisions.  The function that builds the ParameterSetDescription,
        will throw if this rule is violated.  At the moment, the criteria
        for a collision is matching types between a parameter and a wildcard.
        (This criteria is overrestrictive.  With some additional CPU and
        code development the restriction could be loosened to parameters that
        might be injected cannot match the type, trackiness, and wildcard label
        pattern of any wildcard that requires a match.  And further this
        could not apply to wildcards on different branches of a ParameterSwitch
        or "exclusive or".)

    These restrictions have the additional benefit that the things they prohibit
    would tend to confuse a user trying to configure a module or a module developer
    writing the code to extract the parameters from a ParameterSet.  These rules
    tend to prohibit bad design.

    One strategy to avoid problems with wildcard parameters is to add a nested
    ParameterSet and put the wildcard parameters in the nested ParameterSet.
    The names and types in a nested ParameterSet will not interfere with names
    in the containing ParameterSet.
    */
    void checkAndGetLabelsAndTypes(std::set<std::string>& usedLabels,
                                   std::set<ParameterTypes>& parameterTypes,
                                   std::set<ParameterTypes>& wildcardTypes) const {
      checkAndGetLabelsAndTypes_(usedLabels, parameterTypes, wildcardTypes);
    }

    static void printSpaces(std::ostream& os, int n);

  protected:

    virtual void checkAndGetLabelsAndTypes_(std::set<std::string>& usedLabels,
                                            std::set<ParameterTypes>& parameterTypes,
                                            std::set<ParameterTypes>& wildcardTypes) const = 0;

    virtual void validate_(ParameterSet& pset,
                           std::set<std::string>& validatedLabels,
                           bool optional) const = 0;

    virtual void writeCfi_(std::ostream& os,
                           bool& startWithComma,
                           int indentation,
                           bool& wroteSomething) const = 0;

    virtual void print_(std::ostream&,
                        bool /*optional*/,
                        bool /*writeToCfi*/,
                        DocFormatHelper&) { }

    virtual bool hasNestedContent_() {
      return false;
    }

    virtual void printNestedContent_(std::ostream&,
                                     bool /*optional*/,
                                     DocFormatHelper&) { }

    virtual bool exists_(ParameterSet const& pset) const = 0;

    virtual bool partiallyExists_(ParameterSet const& pset) const = 0;

    virtual int howManyXORSubNodesExist_(ParameterSet const& pset) const = 0;

    std::string comment_;
  };

  template <>
  struct value_ptr_traits<ParameterDescriptionNode> {
    static ParameterDescriptionNode* clone(ParameterDescriptionNode const* p) { return p->clone(); }
  };

  // operator>> ---------------------------------------------

  std::auto_ptr<ParameterDescriptionCases<bool> >
  operator>>(bool caseValue,
             ParameterDescriptionNode const& node);

  std::auto_ptr<ParameterDescriptionCases<int> >
  operator>>(int caseValue,
             ParameterDescriptionNode const& node);

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator>>(std::string const& caseValue,
             ParameterDescriptionNode const& node);

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator>>(char const* caseValue,
             ParameterDescriptionNode const& node);

  std::auto_ptr<ParameterDescriptionCases<bool> >
  operator>>(bool caseValue,
             std::auto_ptr<ParameterDescriptionNode> node);

  std::auto_ptr<ParameterDescriptionCases<int> >
  operator>>(int caseValue,
             std::auto_ptr<ParameterDescriptionNode> node);

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator>>(std::string const& caseValue,
             std::auto_ptr<ParameterDescriptionNode> node);

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator>>(char const* caseValue,
             std::auto_ptr<ParameterDescriptionNode> node);

  // operator&& ---------------------------------------------

  std::auto_ptr<ParameterDescriptionNode>
  operator&&(ParameterDescriptionNode const& node_left,
             ParameterDescriptionNode const& node_right);

  std::auto_ptr<ParameterDescriptionNode>
  operator&&(std::auto_ptr<ParameterDescriptionNode> node_left,
             ParameterDescriptionNode const& node_right);

  std::auto_ptr<ParameterDescriptionNode>
  operator&&(ParameterDescriptionNode const& node_left,
             std::auto_ptr<ParameterDescriptionNode> node_right);

  std::auto_ptr<ParameterDescriptionNode>
  operator&&(std::auto_ptr<ParameterDescriptionNode> node_left,
             std::auto_ptr<ParameterDescriptionNode> node_right);

  // operator|| ---------------------------------------------

  std::auto_ptr<ParameterDescriptionNode>
  operator||(ParameterDescriptionNode const& node_left,
             ParameterDescriptionNode const& node_right);

  std::auto_ptr<ParameterDescriptionNode>
  operator||(std::auto_ptr<ParameterDescriptionNode> node_left,
             ParameterDescriptionNode const& node_right);

  std::auto_ptr<ParameterDescriptionNode>
  operator||(ParameterDescriptionNode const& node_left,
             std::auto_ptr<ParameterDescriptionNode> node_right);

  std::auto_ptr<ParameterDescriptionNode>
  operator||(std::auto_ptr<ParameterDescriptionNode> node_left,
             std::auto_ptr<ParameterDescriptionNode> node_right);

  // operator^  ---------------------------------------------

  std::auto_ptr<ParameterDescriptionNode>
  operator^(ParameterDescriptionNode const& node_left,
            ParameterDescriptionNode const& node_right);

  std::auto_ptr<ParameterDescriptionNode>
  operator^(std::auto_ptr<ParameterDescriptionNode> node_left,
            ParameterDescriptionNode const& node_right);

  std::auto_ptr<ParameterDescriptionNode>
  operator^(ParameterDescriptionNode const& node_left,
            std::auto_ptr<ParameterDescriptionNode> node_right);

  std::auto_ptr<ParameterDescriptionNode>
  operator^(std::auto_ptr<ParameterDescriptionNode> node_left,
            std::auto_ptr<ParameterDescriptionNode> node_right);
}
#endif
