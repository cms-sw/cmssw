/**
   \file
   Declaration

   \author Stefano ARGIRO
   \version $Id: UnknownModuleException.h,v 1.2 2005/06/23 22:01:31 wmtan Exp $
   \date 02 Jun 2005
*/

#ifndef _edm_UnknownModuleException_h_
#define _edm_UnknownModuleException_h_

static const char CVSId_edm_UnknownModuleException[] = "$Id: UnknownModuleException.h,v 1.2 2005/06/23 22:01:31 wmtan Exp $";

#include "FWCore/Utilities/interface/Exception.h"

#include <exception>

namespace edm {

  /**
     \class UnknownModuleException UnknownModuleException.h "edm/UnknownModuleException.h"

     \brief Exception thrown when trying to instance a module which is not
            registered to the system

     \author Stefano ARGIRO
     \date 02 Jun 2005
  */
  class UnknownModuleException : public cms::Exception {
  public:

    UnknownModuleException(const std::string & moduletype):
      cms::Exception("UnknownModule")
    {
      (*this) << "Module " << moduletype << " was not registered \n"
	"Perhaps your module type is mispelled or is not a "
	"framework plugin \n"
	"Try running SealPluginDump to obtain a list "
	"of available Plugins\n";
    }
    ~UnknownModuleException() throw(){}
  }; // UnknownModuleException


} // edm


#endif // _edm_UnknownModuleException_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
