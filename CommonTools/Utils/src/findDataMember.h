#ifndef CommonTools_Utils_findDataMember_h
#define CommonTools_Utils_findDataMember_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     findDataMember
// 
/**\class findDataMember findDataMember.h CommonTools/Util/interface/findDataMember.h

 Description: finds a DataMember with a specific name for a Type

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Aug 13 10:07:45 EDT 2008
// $Id: findDataMember.h,v 1.2 2012/08/03 18:08:11 wmtan Exp $
//

// system include files
#include <string>
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
// user include files

// forward declarations
namespace reco {
   edm::MemberWithDict findDataMember(const edm::TypeWithDict& iType, const std::string& iName, int& oError);
}

#endif
