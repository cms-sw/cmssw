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
// $Id: findDataMember.cc,v 1.4 2013/02/10 22:15:06 wmtan Exp $
//

// system include files
#include "FWCore/Utilities/interface/BaseWithDict.h"

// user include files
#include "CommonTools/Utils/src/findDataMember.h"
#include "CommonTools/Utils/src/ErrorCodes.h"

//
// constants, enums and typedefs
//

namespace reco {
   edm::MemberWithDict findDataMember(const edm::TypeWithDict& iType, const std::string& iName, int& oError) {
      edm::MemberWithDict returnValue;
      oError = parser::kNameDoesNotExist;
      edm::TypeWithDict type = iType;
      if(type) {
         if(type.isPointer()) {
            type = type.toType(); // for Pointers, I get the real type this way
         }
         returnValue = type.dataMemberByName(iName);
         if(!returnValue) {
            //check inheriting classes
            edm::TypeBases bases(type);
            for(auto const& base : bases) {
               returnValue = findDataMember(edm::BaseWithDict(base).typeOf(), iName, oError);
               //only stop if we found it or some other error happened
               if(returnValue || parser::kNameDoesNotExist != oError) {
                  break;
               }
            }
         }
         if(returnValue && !returnValue.isPublic()) {
            returnValue = edm::MemberWithDict();
            oError = parser::kIsNotPublic;
         }
      }
      if(returnValue) {
         oError = parser::kNoError;
      }
      return returnValue;
   }
}
