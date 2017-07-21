#ifndef FWCore_Framework_ProductSelectorRules_h
#define FWCore_Framework_ProductSelectorRules_h

//////////////////////////////////////////////////////////////////////
//
// Class ProductSelectorRules. Class for rules to select specific products in event.
//
// Author: Bill Tanenbaum, Marc Paterno
//
//////////////////////////////////////////////////////////////////////

#include <iosfwd>
#include <string>
#include <vector>

#include <regex>

namespace edm {
  class BranchDescription;
  class ProductSelector;
  class ParameterSet;
  class ParameterSetDescription;

  class ProductSelectorRules {
  public:
    ProductSelectorRules(ParameterSet const& pset, std::string const& parameterName, std::string const& parameterOwnerName);
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

    static void fillDescription(ParameterSetDescription& desc, char const* parameterName, std::vector<std::string> const& defaultStrings = defaultSelectionStrings());

    static const std::vector<std::string>& defaultSelectionStrings();
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
      std::regex productType_;
      std::regex moduleLabel_;
      std::regex instanceName_;
      std::regex processName_;
    };
  private:
    std::vector<Rule> rules_;
    std::string parameterName_;
    std::string parameterOwnerName_;
    bool keepAll_;
  };

} // namespace edm



#endif
