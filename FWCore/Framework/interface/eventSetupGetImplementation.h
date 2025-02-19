#ifndef Framework_eventSetupGetImplementation_h
#define Framework_eventSetupGetImplementation_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     eventSetupGetImplementation
// 
/**\class eventSetupGetImplementation eventSetupGetImplementation.h FWCore/Framework/interface/eventSetupGetImplementation.h

 Description: decleration of the template function that implements the EventSetup::get method

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Mar 25 16:31:17 EST 2005
//

// system include files

// user include files

// forward declarations
namespace edm {
   class EventSetup;
   namespace eventsetup {
      
      template< class T>
         void eventSetupGetImplementation(EventSetup const &,
                                       T const *&);
   }
}
#endif
