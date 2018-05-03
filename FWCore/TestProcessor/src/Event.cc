// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     Event
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  root
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
  Event::Event(EventPrincipal const& iPrincipal):
  principal_(&iPrincipal)
{
}


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
