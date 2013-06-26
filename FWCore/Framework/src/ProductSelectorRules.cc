#include <algorithm>
#include <iterator>
#include <ostream>
#include <cctype>

#include "boost/algorithm/string.hpp"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

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
  // (really a ProductHolder, as described by the BranchDescription object
  // that specifies that ProductHolder) matches a 'rule' specified by the
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

  ProductSelectorRules::Rule::Rule(std::string const& s, std::string const& parameterName, std::string const& owner) :
    selectflag_(),
    productType_(),
    moduleLabel_(),
    instanceName_(),
    processName_()
  {
    if (s.size() < 6)
      throw edm::Exception(edm::errors::Configuration)
        << "Invalid statement in configuration file\n"
        << "In " << owner << " parameter named '" << parameterName << "'\n"
        << "Rule must have at least 6 characters because it must\n"
        << "specify 'keep ' or 'drop ' and also supply a pattern.\n"
	<< "This is the invalid output configuration rule:\n" 
	<< "    " << s << "\n"
        << "Exception thrown from ProductSelectorRules::Rule\n";

    if (s.substr(0,4) == "keep")
      selectflag_ = true;
    else if (s.substr(0,4) == "drop")
      selectflag_ = false;
    else
      throw edm::Exception(edm::errors::Configuration)
        << "Invalid statement in configuration file\n"
        << "In " << owner << " parameter named '" << parameterName << "'\n"
        << "Rule must specify 'keep ' or 'drop ' and also supply a pattern.\n"
	<< "This is the invalid output configuration rule:\n" 
	<< "    " << s << "\n"
        << "Exception thrown from ProductSelectorRules::Rule\n";

    if ( !std::isspace(s[4]) ) {

      throw edm::Exception(edm::errors::Configuration)
        << "Invalid statement in configuration file\n"
        << "In " << owner << " parameter named '" << parameterName << "'\n"
        << "In each rule, 'keep' or 'drop' must be followed by a space\n"
	<< "This is the invalid output configuration rule:\n" 
	<< "    " << s << "\n"
        << "Exception thrown from ProductSelectorRules::Rule\n";
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
        << "In " << owner << " parameter named '" << parameterName << "'\n"
        << "In each rule, after 'keep ' or 'drop ' there must\n"
        << "be a branch specification of the form 'type_label_instance_process'\n"
        << "There must be 4 fields separated by underscores\n"
        << "The fields can only contain alphanumeric characters and the wildcards * or ?\n"
        << "Alternately, a single * is also allowed for the branch specification\n"
	<< "This is the invalid output configuration rule:\n" 
	<< "    " << s << "\n"
        << "Exception thrown from ProductSelectorRules::Rule\n";
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
  ProductSelectorRules::Rule::applyToAll(std::vector<BranchSelectState>& branchstates) const {
    std::vector<BranchSelectState>::iterator it = branchstates.begin();
    std::vector<BranchSelectState>::iterator end = branchstates.end();
    for (; it != end; ++it) applyToOne(it->desc, it->selectMe);
  }

  void
  ProductSelectorRules::applyToAll(std::vector<BranchSelectState>& branchstates) const {
    std::vector<Rule>::const_iterator it = rules_.begin();
    std::vector<Rule>::const_iterator end = rules_.end();
    for (; it != end; ++it) it->applyToAll(branchstates);
  }

//   bool
//   Rule::applyToOne(edm::BranchDescription const* branch) const
//   {
//     bool match = 
//       partial_match(productType_, branch->friendlyClassName()) && 
//       partial_match(moduleLabel_, branch->moduleLabel()) &&
//       partial_match(instanceName_, branch->productInstanceName()) &&
//       partial_match(processName_, branch->processName());

//     return match ? selectflag_ : !selectflag_;      
//   }

  void
  ProductSelectorRules::Rule::applyToOne(edm::BranchDescription const* branch,
		   bool& result) const
  {
    if (this->appliesTo(branch)) result = selectflag_;    
  }

  bool
  ProductSelectorRules::Rule::appliesTo(edm::BranchDescription const* branch) const
  {
    return
      partial_match(productType_, branch->friendlyClassName()) && 
      partial_match(moduleLabel_, branch->moduleLabel()) &&
      partial_match(instanceName_, branch->productInstanceName()) &&
      partial_match(processName_, branch->processName());
  }

  void
  ProductSelectorRules::fillDescription(ParameterSetDescription& desc, char const* parameterName) {
    std::vector<std::string> defaultStrings(1U, std::string("keep *"));
    desc.addUntracked<std::vector<std::string> >(parameterName, defaultStrings)
        ->setComment("Specifies which branches are kept or dropped.");
  }

  ProductSelectorRules::ProductSelectorRules(ParameterSet const& pset,
			       std::string const& parameterName,
			       std::string const& parameterOwnerName) :
  rules_(),
  parameterName_(parameterName),
  parameterOwnerName_(parameterOwnerName)
  {
    // Fill the rules.
    // If there is no parameter whose name is parameterName_ in the
    // ParameterSet we are given, we use the following default.
    std::vector<std::string> defaultCommands(1U, std::string("keep *"));

    std::vector<std::string> commands = 
      pset.getUntrackedParameter<std::vector<std::string> >(parameterName,
						    defaultCommands);
    if (commands.empty()) {
      commands.push_back(defaultCommands[0]);
    }
    rules_.reserve(commands.size());
    for(std::vector<std::string>::const_iterator it = commands.begin(), end = commands.end();
        it != end; ++it) {
      rules_.push_back(Rule(*it, parameterName, parameterOwnerName));
    }
    keepAll_ = commands.size() == 1 && commands[0] == defaultCommands[0];
  }


}
