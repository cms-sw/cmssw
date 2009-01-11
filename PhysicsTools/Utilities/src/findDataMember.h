#ifndef PhysicsTools_Utilities_findDataMember_h
#define PhysicsTools_Utilities_findDataMember_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     findDataMember
// 
/**\class findDataMember findDataMember.h PhysicsTools/Utilities/interface/findDataMember.h

 Description: finds a DataMember with a specific name for a Reflex Type

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Aug 13 10:07:45 EDT 2008
// $Id: findDataMember.h,v 1.1 2008/08/13 19:38:18 chrjones Exp $
//

// system include files
#include <string>
#include "Reflex/Member.h"
#include "Reflex/Type.h"
// user include files

// forward declarations
namespace reco {
   Reflex::Member findDataMember(const Reflex::Type& iType, const std::string& iName, int& oError);
}

#endif
