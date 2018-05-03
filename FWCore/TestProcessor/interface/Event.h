#ifndef Subsystem_Package_Event_h
#define Subsystem_Package_Event_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     Event
// 
/**\class Event Event.h "Event.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Mon, 30 Apr 2018 18:51:27 GMT
//

// system include files

// user include files

// forward declarations

namespace edm {
  class EventPrincipal;
  
namespace test {
  
class Event
{

   public:
      Event(EventPrincipal const& iPrincipal);

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:

      // ---------- member data --------------------------------
  EventPrincipal const* principal_;
};
}
}


#endif
