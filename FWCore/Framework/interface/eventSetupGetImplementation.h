#ifndef EVENTSETUP_EVENTSETUPGETIMPLEMENTATION_H
#define EVENTSETUP_EVENTSETUPGETIMPLEMENTATION_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     eventSetupGetImplementation
// 
/**\class eventSetupGetImplementation eventSetupGetImplementation.h Core/CoreFramework/interface/eventSetupGetImplementation.h

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
#endif /* EVENTSETUP_EVENTSETUPGETIMPLEMENTATION_H */
