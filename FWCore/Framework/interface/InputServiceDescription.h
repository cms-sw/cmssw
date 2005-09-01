#ifndef Framework_InputServiceDescription_h
#define Framework_InputServiceDescription_h

/*----------------------------------------------------------------------

InputServiceDescription : the stuff that is needed to configure an
input service that does not come in through the ParameterSet  

$Id: InputServiceDescription.h,v 1.5 2005/07/30 23:44:24 wmtan Exp $
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
