#ifndef INPUTSERVICEDESCRIPTION_H
#define INPUTSERVICEDESCRIPTION_H

/*----------------------------------------------------------------------

InputServiceDescription : the stuff that is needed to configure an
input service that does not come in through the ParameterSet  

$Id: InputServiceDescription.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include "FWCore/Framework/interface/PassID.h"

namespace edm
{
  struct InputServiceDescription
  {
    InputServiceDescription() : process_name(), pass() { }
    InputServiceDescription(const std::string& name, PassID pid) :
      process_name(name),
      pass(pid)
    {}

    std::string process_name;
    PassID      pass;
  };
}

#endif
