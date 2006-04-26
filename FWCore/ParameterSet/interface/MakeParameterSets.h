#ifndef FWCoreParameterSet_MakeParameterSets_h
#define FWCoreParameterSet_MakeParameterSets_h


//----------------------------------------------------------------------
// Declare functions used to create ParameterSets.
//
// $Id: MakeParameterSets.h,v 1.3 2006/01/17 23:22:45 paterno Exp $
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




} // namespace edm

#endif
