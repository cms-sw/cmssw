// $Id: GroupSelector.cc,v 1.25 2007/06/29 03:43:21 wmtan Exp $

#include <algorithm>
#include <iterator>
#include <ostream>
#include <cctype>

#include "boost/algorithm/string.hpp"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"


namespace edm {
// The following typedef is used only in this implementation file, in
// order to shorten several lines of code.
typedef std::vector<edm::BranchDescription const*> VCBDP;

  namespace 
  {
  
    //--------------------------------------------------
    // function partial_match is a helper for Rule. It encodes the
    // matching of std::strings, and knows about wildcarding rules.
    inline
    bool
    partial_match(const boost::regex& regularExpression,
  		  const std::string& branchstring)
    {
      if (regularExpression.empty()) {
        if (branchstring == "") return true;
        else return false;
      }
      return boost::regex_match(branchstring, regularExpression);
    }
  }
  //--------------------------------------------------  
  // Class Rule is used to determine whether or not a given branch
  // (really a Group, as described by the BranchDescription object
  // that specifies that Group) matches a 'rule' specified by the
  // configuration. Each Rule is configured with a single std::string from
  // the configuration file.
  //
  // The configuration std::string is of the form:
  //
  //   'keep <spec>'            ** or **
  //   'drop <spec>'
  //
  // where '<spec>' is of the form:
  //
  //   <product type>_<module label>_<instance name>_<process name>
  //
  // The 3 underscores must always be present.  The four fields can
  // be empty or composed of alphanumeric characters.  "*" is an
  // allowed wildcard that will match 0 or more of any characters.
  // "?" is the other allowed wilcard that will match exactly one
  // character.  There is one exception to this, the entire '<spec>'
  // can be one single "*" without any underscores and this is
  // interpreted as "*_*_*_*".  Anything else will lead to an exception
  // being thrown.
  //
  // This class has much room for optimization. This should be
  // revisited as soon as profiling data are available.

  GroupSelector::Rule::Rule(std::string const& s) :
    writeflag_(),
    productType_(),
    moduleLabel_(),
    instanceName_(),
    processName_()
  {
    if (s.size() < 6)
      throw edm::Exception(edm::errors::Configuration)
        << "Invalid statement in configuration file\n"
        << "In OutputModule parameter named 'outputCommands'\n"
        << "Rule must have at least 6 characters because it must\n"
        << "specify 'keep ' or 'drop ' and also supply a pattern.\n"
	<< "This is the invalid output configuration rule:\n" 
	<< "    " << s << "\n"
        << "Exception thrown from GroupSelector::Rule::Rule\n";

    if (s.substr(0,4) == "keep")
      writeflag_ = true;
    else if (s.substr(0,4) == "drop")
      writeflag_ = false;
    else
      throw edm::Exception(edm::errors::Configuration)
        << "Invalid statement in configuration file\n"
        << "In OutputModule parameter named 'outputCommands'\n"
        << "Rule must specify 'keep ' or 'drop ' and also supply a pattern.\n"
	<< "This is the invalid output configuration rule:\n" 
	<< "    " << s << "\n"
        << "Exception thrown from GroupSelector::Rule::Rule\n";

    if ( !std::isspace(s[4]) ) {

      throw edm::Exception(edm::errors::Configuration)
        << "Invalid statement in configuration file\n"
        << "In OutputModule parameter named 'outputCommands'\n"
        << "In each rule, 'keep' or 'drop' must be followed by a space\n"
	<< "This is the invalid output configuration rule:\n" 
	<< "    " << s << "\n"
        << "Exception thrown from GroupSelector::Rule::Rule\n";
    }

    // Now pull apart the std::string to get at the bits and pieces of the
    // specification...
    
    // Grab from after 'keep/drop ' (note the space!) to the end of
    // the std::string...
    std::string spec(s.begin()+5, s.end());

    // Trim any leading and trailing whitespace from spec
    boost::trim(spec);

    if (spec == "*") // special case for wildcard
    {
      productType_  = ".*";
      moduleLabel_  = ".*";
      instanceName_ = ".*";
      processName_  = ".*";
      return;
    }
    else
    {
      std::vector<std::string> parts;
      boost::split(parts, spec, boost::is_any_of("_"));

      // The std::vector must contain at least 4 parts
      // and none may be empty.
      bool good = (parts.size() == 4);

      // Require all the std::strings to contain only alphanumberic
      // characters or "*" or "?"
      if (good) 
      {
        for (int i = 0; i < 4; ++i) {
	  std::string& field = parts[i];
          int size = field.size();
          for (int j = 0; j < size; ++j) {
            if ( !(isalnum(field[j]) || field[j] == '*' || field[j] == '?') ) {
              good = false;
            }
          }

          // We are using the boost regex library to deal with the wildcards.
          // The configuration file uses a syntax that accepts "*" and "?"
          // as wildcards so we need to convert these to the syntax used in
          // regular expressions.
          boost::replace_all(parts[i], "*", ".*");
          boost::replace_all(parts[i], "?", ".");
        }
      }

      if (!good)
      {
      throw edm::Exception(edm::errors::Configuration)
        << "Invalid statement in configuration file\n"
        << "In OutputModule parameter named 'outputCommands'\n"
        << "In each rule, after 'keep ' or 'drop ' there must\n"
        << "be a branch specification of the form 'type_label_instance_process'\n"
        << "There must be 4 fields separated by underscores\n"
        << "The fields can only contain alphanumeric characters and the wildcards * or ?\n"
        << "Alternately, a single * is also allowed for the branch specification\n"
	<< "This is the invalid output configuration rule:\n" 
	<< "    " << s << "\n"
        << "Exception thrown from GroupSelector::Rule::Rule\n";
      }

      // Assign the std::strings to the regex (regular expression) objects
      // If the std::string is empty we skip the assignment and leave
      // the regular expression also empty.

      if (parts[0] != "") productType_  = parts[0];
      if (parts[1] != "") moduleLabel_  = parts[1];
      if (parts[2] != "") instanceName_ = parts[2];
      if (parts[3] != "") processName_  = parts[3];
    }
  }

  void
  GroupSelector::Rule::applyToAll(std::vector<GroupSelector::BranchWriteState>& branchstates) const
  {
    std::vector<GroupSelector::BranchWriteState>::iterator it = branchstates.begin();
    std::vector<GroupSelector::BranchWriteState>::iterator end = branchstates.end();
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
    std::vector<std::string> defaultCommands(1U, std::string("keep *"));

    std::vector<std::string> commands = 
      params.getUntrackedParameter<std::vector<std::string> >("outputCommands",
						    defaultCommands);
    rules_.reserve(commands.size());
    std::vector<std::string>::const_iterator it =  commands.begin();
    std::vector<std::string>::const_iterator end = commands.end();
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
    return binary_search_all(groupsToWrite_, desc.branchName());
  }

  void
  GroupSelector::print(std::ostream& os) const
  {
    os << "GroupSelector at: "
       << static_cast<void const*>(this)
       << " has "
       << groupsToWrite_.size()
       << " groups to write:\n";      
    copy_all(groupsToWrite_, std::ostream_iterator<std::string>(os, "\n"));
  }


  void GroupSelector::initialize(VCBDP const& allBranches)
  {

    // Get a BranchWriteState for each branch, containing the branch
    // name, with its 'write bit' set to false.
    std::vector<GroupSelector::BranchWriteState> branchstates;
    {
      branchstates.reserve(allBranches.size());
      
      VCBDP::const_iterator it = allBranches.begin();
      VCBDP::const_iterator end = allBranches.end();
      for (; it != end; ++it) branchstates.push_back(GroupSelector::BranchWriteState(*it));
    }

    // Now  apply the rules to  the branchstates, in order.  Each rule
    // can override any previous rule, or all previous rules.
    {
      std::vector<GroupSelector::Rule>::const_iterator it = rules_.begin();
      std::vector<GroupSelector::Rule>::const_iterator end = rules_.end();
      for (; it != end; ++it) it->applyToAll(branchstates);
    }

    // For each of the BranchWriteStates that indicates the branch is
    // to be written, remember the branch name.  The list of branch
    // names must be sorted, for the implementation of 'selected' to
    // work.
    {
      std::vector<GroupSelector::BranchWriteState>::const_iterator it = branchstates.begin();
      std::vector<GroupSelector::BranchWriteState>::const_iterator end = branchstates.end();
      for (; it != end; ++it) {
	  if (it->writeMe) groupsToWrite_.push_back(it->desc->branchName());
      }
      sort_all(groupsToWrite_);
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
