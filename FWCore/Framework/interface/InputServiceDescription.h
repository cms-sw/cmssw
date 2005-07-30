#ifndef INPUTSERVICEDESCRIPTION_H
#define INPUTSERVICEDESCRIPTION_H

/*----------------------------------------------------------------------

InputServiceDescription : the stuff that is needed to configure an
input service that does not come in through the ParameterSet  

$Id: InputServiceDescription.h,v 1.4 2005/07/23 05:19:44 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include "FWCore/Framework/interface/PassID.h"

namespace edm {
  class ProductRegistry;

  struct InputServiceDescription {
    InputServiceDescription() : processName_(), pass(), preg_(0) { }
    InputServiceDescription(std::string const& name, PassID pid, 
			    ProductRegistry& preg) :
      processName_(name),
      pass(pid),
      preg_(&preg)
	 
    {}

    std::string const processName_;
    PassID      pass;
    ProductRegistry * preg_;
  };
}

#endif
