#ifndef FWCore_Framework_GroupSelectorRules_h
#define FWCore_Framework_GroupSelectorRules_h

//////////////////////////////////////////////////////////////////////
//
// $Id: GroupSelectorRules.h,v 1.1 2008/06/05 23:17:05 wmtan Exp $
//
// Class GroupSelectorRules. Class for rules to select specific groups in event.
//
// Author: Bill Tanenbaum, Marc Paterno
//
//////////////////////////////////////////////////////////////////////

#include <iosfwd>
#include <string>
#include <vector>

#include <boost/regex.hpp>

namespace edm {
  class BranchDescription;
  class GroupSelector;
  class ParameterSet;

  class GroupSelectorRules {
  public:
    GroupSelectorRules(ParameterSet const& pset, std::string const& parameterName, std::string const& parameterOwnerName);
    //--------------------------------------------------
    // BranchSelectState is a struct which associates a BranchDescription
    // (*desc) with a bool indicating whether or not the branch with
    // that name is to be selected.  Note that desc may not be null.
    struct BranchSelectState {
      edm::BranchDescription const* desc;
      bool                          selectMe;
  
      // N.B.: We assume bd is not null.
      explicit BranchSelectState (edm::BranchDescription const* bd) : 
        desc(bd), 
        selectMe(false)
      { }
    };
  
    void applyToAll(std::vector<BranchSelectState>& branchstates) const;

    bool keepAll() const {return keepAll_;}

  private:
    class Rule {
    public:
      Rule(std::string const& s, std::string const& parameterName, std::string const& owner);
  
      // Apply the rule to all the given branch states. This may modify
      // the given branch states.
      void applyToAll(std::vector<BranchSelectState>& branchstates) const;
  
      // Apply the rule to the given BranchDescription. The return value
      // is the value to which the 'select bit' should be set, according
      // to application of this rule.
      //bool applyToOne(BranchDescription const* branch) const;
  
      // If this rule applies to the given BranchDescription, then
      // modify 'result' to match the rule's select flag. If the rule does
      // not apply, do not modify 'result'.
      void applyToOne(BranchDescription const* branch, bool& result) const;
  
      // Return the answer to the question: "Does the rule apply to this
      // BranchDescription?"
      bool appliesTo(BranchDescription const* branch) const;
  
    private:
      // selectflag_ carries the value to which we should set the 'select
      // bit' if this rule matches.
      bool   selectflag_;
      boost::regex productType_;
      boost::regex moduleLabel_;
      boost::regex instanceName_;
      boost::regex processName_;
    };
  private:
    std::vector<Rule> rules_;
    std::string parameterName_;
    std::string parameterOwnerName_;
    bool keepAll_;
  };

} // namespace edm



#endif
