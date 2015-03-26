// -*- C++ -*-
//
// Package:     Utilities
// Class  :     findDataMember
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Aug 13 10:07:46 EDT 2008
// $Id: findDataMember.cc,v 1.3 2012/08/03 18:08:11 wmtan Exp $
//

// system include files
#include "TInterpreter.h"
#include "TVirtualMutex.h"

#include "FWCore/Utilities/interface/BaseWithDict.h"

// user include files
#include "CommonTools/Utils/src/findDataMember.h"
#include "CommonTools/Utils/src/ErrorCodes.h"

//
// constants, enums and typedefs
//

namespace reco {

edm::
MemberWithDict
findDataMember(const edm::TypeWithDict& iType,
               const std::string& iName,
               int& oError)
{
  edm::MemberWithDict ret;
  oError = parser::kNameDoesNotExist;
  edm::TypeWithDict type = iType;
  if (!bool(type)) {
    return ret;
  }
  if (type.isPointer()) {
    type = type.toType();
  }
  ret = type.dataMemberByName(iName);
  if (!bool(ret)) {
    // check base classes
    edm::TypeBases bases(type);
    for (auto const& B : bases) {
      ret = findDataMember(edm::BaseWithDict(B).typeOf(), iName, oError);
      //only stop if we found it or some other error happened
      if (bool(ret) || (oError != parser::kNameDoesNotExist)) {
        break;
      }
    }
  }
  if (bool(ret) && !ret.isPublic()) {
    ret = edm::MemberWithDict();
    oError = parser::kIsNotPublic;
  }
  else if (bool(ret)) {
    oError = parser::kNoError;
  }
  return ret;
}

} // namespace reco

