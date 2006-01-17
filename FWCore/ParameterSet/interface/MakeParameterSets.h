#ifndef FWCoreParameterSet_MakeParameterSets_h
#define FWCoreParameterSet_MakeParameterSets_h


//----------------------------------------------------------------------
// Declare functions used to create ParameterSets.
//
// $Id: MakeParameterSets.h,v 1.2 2006/01/17 20:32:31 paterno Exp $
//
//----------------------------------------------------------------------

#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm
{

  // Parse the given configuration text, filling in as output:
  //
  //   1. main, the ParameterSet to be used to configure the
  //      EventProcessor, and
  //   2. servicesparams, a vector of parameter sets to be used
  //      to configure services.
  //
  // Both of the output arguments will be over-written.
  // Failure of parsing results in an exception.

  void 
  makeParameterSets(std::string const& configtext,
		    boost::shared_ptr<ParameterSet>& main,
		    boost::shared_ptr<std::vector<ParameterSet> >& serviceparams);


  // The following are implementation details. The prototypes are here
  // so that the functions may be tested.
  namespace pset
  {
    bool read_whole_file(std::string const& filename, std::string& output);
    bool is_include_line(std::string const& input, std::string& filename);
    void preprocessConfigString(std::string const& input,
				std::string& output,
				std::vector<std::string>& openFileStack);    
  }


} // namespace edm

#endif
