#ifndef CommonTools_Utils_findDataMember_h
#define CommonTools_Utils_findDataMember_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     findDataMember
// 
/**\class findDataMember findDataMember.h CommonTools/Util/interface/findDataMember.h

 Description: finds a DataMember with a specific name for a Reflex Type

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Aug 13 10:07:45 EDT 2008
// $Id: findDataMember.h,v 1.2 2009/01/11 23:37:33 hegner Exp $
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
