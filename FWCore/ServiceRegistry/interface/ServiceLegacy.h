#ifndef ServiceRegistry_ServiceLegacy_h
#define ServiceRegistry_ServiceLegacy_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceLegacy
// 
/**\class ServiceLegacy ServiceLegacy.h FWCore/ServiceRegistry/interface/ServiceLegacy.h

 Description: Enumeration of how Services inherit from other Service sets

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Sep  7 13:42:29 EDT 2005
// $Id: ServiceLegacy.h,v 1.1 2005/09/07 21:58:15 chrjones Exp $
//

// system include files

// user include files

// forward declarations

namespace edm {
   namespace serviceregistry {
      enum ServiceLegacy {
         kOverlapIsError,
         kTokenOverrides,
         kConfigurationOverrides
      };
   }
}

#endif
