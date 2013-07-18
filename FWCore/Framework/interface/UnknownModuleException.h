#ifndef FWCore_Framework_UnknownModuleException_h
#define FWCore_Framework_UnknownModuleException_h

/**
   \file
   Declaration

   \author Stefano ARGIRO
   \version $Id: UnknownModuleException.h,v 1.7 2007/06/14 17:52:16 wmtan Exp $
   \date 02 Jun 2005
*/

#include "FWCore/Utilities/interface/Exception.h"

#include <string>

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
	"Perhaps your module type is misspelled or is not a "
	"framework plugin \n"
	"Try running EdmPluginDump to obtain a list "
	"of available Plugins\n";
    }
    ~UnknownModuleException() throw(){}
  }; // UnknownModuleException


} // edm

#endif
