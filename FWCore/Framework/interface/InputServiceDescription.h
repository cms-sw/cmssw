#ifndef INPUTSERVICEDESCRIPTION_H
#define INPUTSERVICEDESCRIPTION_H

/*----------------------------------------------------------------------

InputServiceDescription : the stuff that is needed to configure an
input service that does not come in through the ParameterSet  

$Id: InputServiceDescription.h,v 1.3 2005/07/21 20:45:47 argiro Exp $
----------------------------------------------------------------------*/
#include <string>
#include "FWCore/Framework/interface/PassID.h"

namespace edm {
  class ProductRegistry;

  struct InputServiceDescription {
    InputServiceDescription() : process_name(), pass(), preg_(0) { }
    InputServiceDescription(std::string const& name, PassID pid, 
			    ProductRegistry& preg) :
      process_name(name),
      pass(pid),
      preg_(&preg)
	 
    {}

    std::string const process_name;
    PassID      pass;
    ProductRegistry * preg_;
  };
}

#endif
