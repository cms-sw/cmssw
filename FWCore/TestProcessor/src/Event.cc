// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     Event
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Mon, 30 Apr 2018 18:51:33 GMT
//

// system include files

// user include files
#include "FWCore/TestProcessor/interface/Event.h"

namespace edm {
namespace test {
    
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
  Event::Event(EventPrincipal const& iPrincipal, std::string iModuleLabel, std::string iProcessName,
               bool modulePassed):
  principal_{&iPrincipal},
  label_{std::move(iModuleLabel)},
  processName_{std::move(iProcessName)},
  modulePassed_(modulePassed){}


//
// member functions
//

//
// const member functions
//

//
// static member functions
//

}
}
