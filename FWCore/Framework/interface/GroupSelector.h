#ifndef Framework_GroupSelector_h
#define Framework_GroupSelector_h

//////////////////////////////////////////////////////////////////////
//
// $Id: GroupSelector.h,v 1.9 2006/01/11 00:21:31 paterno Exp $
//
// Class GroupSelector. Class for user to select specific groups in event.
//
// Author: Bill Tanenbaum, Marc Paterno
//
//////////////////////////////////////////////////////////////////////

#include <iosfwd>
#include <string>
#include <vector>


namespace edm {
  class BranchDescription;
  class ParameterSet;

  class GroupSelector {
  public:
    // N.B.: we assume there are not null pointers in the vector allBranches.
    explicit GroupSelector(ParameterSet const& ps);

    void initialize(std::vector<BranchDescription const*> const& 
		     branchDescriptions);

    bool selected(BranchDescription const& desc) const;

    // Printout intended for debugging purposes.
    void print(std::ostream& os) const;

    bool initialized() const {return initialized_;}

  private:

    //--------------------------------------------------
    // BranchWriteState is a struct which associates a BranchDescription
    // (*desc) with a bool indicating whether or not the branch with
    // that name is to be written.  Note that desc may not be null.
    struct BranchWriteState
    {
      edm::BranchDescription const* desc;
      bool                          writeMe;
  
      // N.B.: We assume bd is not null.
      explicit BranchWriteState (edm::BranchDescription const* bd) : 
        desc(bd), 
        writeMe(false)
      { }
    };
  
    class Rule
    {
    public:
      explicit Rule(std::string const& s);
  
      // Apply the rule to all the given branch states. This may modify
      // the given branch states.
      void applyToAll(std::vector<GroupSelector::BranchWriteState>& branchstates) const;
  
      // Apply the rule to the given BranchDescription. The return value
      // is the value to which the 'write bit' should be set, according
      // to application of this rule.
      //bool applyToOne(edm::BranchDescription const* branch) const;
  
      // If this rule applies to the given BranchDescription, then
      // modify 'result' to match the rule's write flag. If the rule does
      // not apply, do not modify 'result'.
      void applyToOne(edm::BranchDescription const* branch,
  		    bool& result) const;
  
      // Return the answer to the question: "Does the rule apply to this
      // BranchDescription?"
      bool appliesTo(edm::BranchDescription const* branch) const;
  
    private:
  
  
      // writeflag_ carries the value to which we should set the 'write
      // bit' if this rule matches.
      bool   writeflag_;
      std::string productType_;
      std::string moduleLabel_;
      std::string instanceName_;
      std::string processName_;
    };

   void fill_rules(edm::ParameterSet const& params);
  
    // We keep a sorted collection of branch names, indicating the
    // groups which are to be written.

    // TODO: See if we can keep pointer to (const) BranchDescriptions,
    // so that we can do pointer comparison rather than string
    // comparison. This will work if the BranchDescription we are
    // given in the 'selected' member function is one of the instances
    // that are managed by the ProductRegistry used to initialize the
    // OutputModule that contains this GroupSelector.
    std::vector<std::string> groupsToWrite_;
    std::vector<Rule> rules_;
    bool initialized_;
  };

  std::ostream&
  operator<< (std::ostream& os, const GroupSelector& gs);

} // namespace edm



#endif
