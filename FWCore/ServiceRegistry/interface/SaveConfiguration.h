#ifndef FWCore_ServiceRegistry_SaveConfiguration_h
#define FWCore_ServiceRegistry_SaveConfiguration_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     SaveConfiguration
// 
/**\class SaveConfiguration SaveConfiguration.h FWCore/ServiceRegistry/interface/SaveConfiguration.h

 Description: 'Concept' class used to decide if a Service's parameters should be saved

 Usage:
    Inherit from this class if and only if you wish your Service's parameters to be stored to the
 EDM ROOT files provenance.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Mar 12 14:55:07 CST 2010
// $Id$
//

// system include files

// user include files

// forward declarations

namespace edm {
   namespace serviceregistry {
      class SaveConfiguration
      {
      };
   }
}

#endif
