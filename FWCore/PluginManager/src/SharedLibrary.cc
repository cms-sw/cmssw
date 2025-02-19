// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     SharedLibrary
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Apr  5 15:30:15 EDT 2007
// $Id: SharedLibrary.cc,v 1.4 2011/08/24 12:28:53 eulisse Exp $
//

// system include files
#include <string> /*needed by the following include*/
#include "Reflex/SharedLibrary.h"

// user include files
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edmplugin {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
  SharedLibrary::SharedLibrary(const boost::filesystem::path& iName) :
  library_(0),
  path_(iName)
{
    std::auto_ptr<Reflex::SharedLibrary> lib(new Reflex::SharedLibrary(iName.string()));
    if( !lib->Load() ) {
      throw cms::Exception("PluginLibraryLoadError")<<"unable to load "<<iName.string()<<" because "<<lib->Error();
    }
    library_ = lib.release();
}

// SharedLibrary::SharedLibrary(const SharedLibrary& rhs)
// {
//    // do actual copying here;
// }

SharedLibrary::~SharedLibrary()
{
  delete library_;
}

//
// assignment operators
//
// const SharedLibrary& SharedLibrary::operator=(const SharedLibrary& rhs)
// {
//   //An exception safe implementation is
//   SharedLibrary temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
bool 
SharedLibrary::symbol(const std::string& iSymbolName, void* & iSymbol) const
{
  return library_->Symbol(iSymbolName,iSymbol);
}

//
// static member functions
//
}
