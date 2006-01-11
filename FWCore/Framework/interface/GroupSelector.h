#ifndef Framework_GroupSelector_h
#define Framework_GroupSelector_h

//////////////////////////////////////////////////////////////////////
//
// $Id: GroupSelector.h,v 1.8 2005/12/28 00:14:52 wmtan Exp $
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
    GroupSelector(ParameterSet const& ps,
		  std::vector<BranchDescription const*> const& allBranches);

    bool selected(BranchDescription const& desc) const;

    // Printout intended for debugging purposes.
    void print(std::ostream& os) const;

  private:
    void initialize_(ParameterSet const& pset,
		     std::vector<BranchDescription const*> const& 
		     branchDescriptions);

    // We keep a sorted collection of branch names, indicating the
    // groups which are to be written.

    // TODO: See if we can keep pointer to (const) BranchDescriptions,
    // so that we can do pointer comparison rather than string
    // comparison. This will work if the BranchDescription we are
    // given in the 'selected' member function is one of the instances
    // that are managed by the ProductRegistry used to initialize the
    // OutputModule that contains this GroupSelector.
    std::vector<std::string> groupsToWrite_;
  };

  std::ostream&
  operator<< (std::ostream& os, const GroupSelector& gs);

} // namespace edm



#endif
