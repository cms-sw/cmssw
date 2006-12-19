// $Id: GroupSelector.cc,v 1.20 2006/08/31 23:26:24 wmtan Exp $

#include <algorithm>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"

#include "DataFormats/Common/interface/BranchDescription.h"
#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
//#include "FWCore/Framework/interface/ConstProductRegistry.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"

using std::string;
using std::vector;


namespace edm {
// The following typedef is used only in this implementation file, in
// order to shorten several lines of code.
typedef std::vector<edm::BranchDescription const*> VCBDP;

  namespace 
  {
  
    //--------------------------------------------------
    // function partial_match is a helper for Rule. It encodes the
    // matching of strings, and knows about wildcarding rules.
    // N.B.: an empty rulestring matches *everything*.
    inline
    bool
    partial_match(string const& rulestring,
  		string const& branchstring)
    {
      return rulestring.empty() ? true : rulestring == branchstring;    
    }
  
  }
  //--------------------------------------------------  
  // Class Rule is used to determine whether or not a given branch
  // (really a Group, as described by the BranchDescription object
  // that specifes that Group) matches a 'rule' specified by the
  // configuration. Each Rule is configured with a single string from
  // the configuration file.
  //
  // The configuration string is of the form:
  //
  //   'keep <spec>'            ** or **
  //   'drop <spec>'
  //
  // where '<spec>' is of the form:
  //
  //   <product type>_<module label>_<instance name>_<process name>
  //
  // with the abbreviations allowed branch names (see
  // FWCore/Framework/src/BranchDescription.cc for details). The
  // wildcard '*' is used to indicate that all values are to be
  // matched in the field in which the wildcard appears.  The full
  // four-field pattern must be specified, except in the special case
  // of the configuration string '*', which is converted to '*_*_*_*',
  // which matches everything.
  //
  // This class has much room for optimization. This should be
  // revisited as soon as profiling data are available.

  GroupSelector::Rule::Rule(string const& s) :
    writeflag_(),
    productType_(),
    moduleLabel_(),
    instanceName_(),
    processName_()
  {
    // Configuration strings are of the form:
    //   'keep|drop  T_M_U_P'     or
    //   'keep|drop  *'
    //
    if (s.size() < 6)
      throw edm::Exception(edm::errors::Configuration)
	<< "Command must specify 'keep' or 'drop',  and supply a pattern"
	<< "to match\n"
	<< "invalid output configuration rule: " 
	<< s;

    if (s.substr(0,4) == "keep")
      writeflag_ = true;
    else if (s.substr(0,4) == "drop")
      writeflag_ = false;
    else
      throw edm::Exception(edm::errors::Configuration,
			   "Command must specify 'keep' or 'drop'")
	<< "invalid output configuration rule: " 
	<< s;    

    // Now pull apart the string to get at the bits and pieces of the
    // specification...
    
    // Grab from after 'keep/drop ' (note the space!) to the end of
    // the string...
    string spec(s.begin()+5, s.end());

    // Trim any leading and trailing whitespace from spec
    boost::trim(spec);

    if (spec == "*") // special case for wildcard
      {
	return; // we're done: all string data members are empty
      }
    else
      {
	vector<string> parts;
	boost::split(parts, spec, boost::is_any_of("_"));

	// The vector must contain at least 4 parts
	// and none may be empty.
	bool good = (parts.size() == 4);
	if (good) 
	  {
	    for (int i = 0; i < 4; ++i) good &= !parts[i].empty();
	  }

	if (!good)
	  {
	    throw edm::Exception(edm::errors::Configuration)
	      << "Branch specification must be either '*'\n"
	      << "or have the four-part form <T>_<M>_<I>_<P>\n"
	      << "invalid output configuration rule: "
	      << s;
	  }
	
	if (parts[0] != "*") productType_  = parts[0];
	if (parts[1] != "*") moduleLabel_  = parts[1];
	if (parts[2] != "*") instanceName_ = parts[2];
	if (parts[3] != "*") processName_  = parts[3];
      }
  }

  void
  GroupSelector::Rule::applyToAll(vector<GroupSelector::BranchWriteState>& branchstates) const
  {
    vector<GroupSelector::BranchWriteState>::iterator it = branchstates.begin();
    vector<GroupSelector::BranchWriteState>::iterator end = branchstates.end();
    for (; it != end; ++it) applyToOne(it->desc, it->writeMe);
  }

//   bool
//   Rule::applyToOne(edm::BranchDescription const* branch) const
//   {
//     bool match = 
//       partial_match(productType_, branch->friendlyClassName()) && 
//       partial_match(moduleLabel_, branch->moduleLabel()) &&
//       partial_match(instanceName_, branch->productInstanceName()) &&
//       partial_match(processName_, branch->processName());

//     return match ? writeflag_ : !writeflag_;      
//   }

  void
  GroupSelector::Rule::applyToOne(edm::BranchDescription const* branch,
		   bool& result) const
  {
    if (this->appliesTo(branch)) result = writeflag_;    
  }

  bool
  GroupSelector::Rule::appliesTo(edm::BranchDescription const* branch) const
  {
    return
      partial_match(productType_, branch->friendlyClassName()) && 
      partial_match(moduleLabel_, branch->moduleLabel()) &&
      partial_match(instanceName_, branch->productInstanceName()) &&
      partial_match(processName_, branch->processName());
  }

  GroupSelector::GroupSelector(ParameterSet const& pset) :
    groupsToWrite_(),
    rules_(),
    initialized_(false)
  {
    // Create a Rule for each command.
    fill_rules(pset);
  }

  //--------------------------------------------------
  // function fill_rules is a helper for GroupSelector's
  // initilaization. It deals with creating rules, as specified in the
  // given ParameterSet object.
  void 
  GroupSelector::fill_rules(edm::ParameterSet const& params)
  {
    // If there is no parameter named 'outputCommands' in the
    // ParameterSet we are given, we use the following default.
    vector<string> defaultCommands(1U, string("keep *"));

    vector<string> commands = 
      params.getUntrackedParameter<vector<string> >("outputCommands",
						    defaultCommands);
    rules_.reserve(commands.size());
    vector<string>::const_iterator it =  commands.begin();
    vector<string>::const_iterator end = commands.end();
    for (; it != end; ++it) {
      rules_.push_back(GroupSelector::Rule(*it));
    }
  }

  bool GroupSelector::selected(BranchDescription const& desc) const 
  {
    if (!initialized_) {
      throw edm::Exception(edm::errors::LogicError)
        << "GroupSelector::selected() called prematurely\n"
        << "before the product registry has been frozen.\n";
    }
    // We are to write this 'branch' if its name is one of the ones we
    // have been told to write.
    return std::binary_search(groupsToWrite_.begin(), 
			       groupsToWrite_.end(),
			       desc.branchName());
  }

  void
  GroupSelector::print(std::ostream& os) const
  {
    os << "GroupSelector at: "
       << static_cast<void const*>(this)
       << " has "
       << groupsToWrite_.size()
       << " groups to write:\n";      
    std::copy(groupsToWrite_.begin(),
	      groupsToWrite_.end(),
	      std::ostream_iterator<string>(os, "\n"));
  }


  void GroupSelector::initialize(VCBDP const& allBranches)
  {

    // Get a BranchWriteState for each branch, containing the branch
    // name, with its 'write bit' set to false.
    vector<GroupSelector::BranchWriteState> branchstates;
    {
      branchstates.reserve(allBranches.size());
      
      VCBDP::const_iterator it = allBranches.begin();
      VCBDP::const_iterator end = allBranches.end();
      for (; it != end; ++it) branchstates.push_back(GroupSelector::BranchWriteState(*it));
    }

    // Now  apply the rules to  the branchstates, in order.  Each rule
    // can override any previous rule, or all previous rules.
    {
      vector<GroupSelector::Rule>::const_iterator it = rules_.begin();
      vector<GroupSelector::Rule>::const_iterator end = rules_.end();
      for (; it != end; ++it) it->applyToAll(branchstates);
    }

    // For each of the BranchWriteStates that indicates the branch is
    // to be written, remember the branch name.  The list of branch
    // names must be sorted, for the implementation of 'selected' to
    // work.
    {
      vector<GroupSelector::BranchWriteState>::const_iterator it = branchstates.begin();
      vector<GroupSelector::BranchWriteState>::const_iterator end = branchstates.end();
      for (; it != end; ++it)
	{
	  if (it->writeMe) groupsToWrite_.push_back(it->desc->branchName());
	}
      std::sort(groupsToWrite_.begin(), groupsToWrite_.end());
    }
    initialized_ = true;
  }

  //--------------------------------------------------
  //
  // Associated free functions
  //
  std::ostream&
  operator<< (std::ostream& os, const GroupSelector& gs)
  {
    gs.print(os);
    return os;
  }
  
}
