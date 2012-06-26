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
// $Id: findDataMember.cc,v 1.1 2009/02/24 14:10:22 llista Exp $
//

// system include files
#include "Reflex/Base.h"

// user include files
#include "CommonTools/Utils/src/findDataMember.h"
#include "CommonTools/Utils/src/ErrorCodes.h"

//
// constants, enums and typedefs
//

namespace reco {
   Reflex::Member findDataMember(const Reflex::Type& iType, const std::string& iName, int& oError) {
      Reflex::Member returnValue;
      oError = parser::kNameDoesNotExist;
      Reflex::Type type = iType;
      if(type) {
         if(type.IsPointer()) {
            type = type.ToType();
         }
         returnValue = type.DataMemberByName(iName);
         if(!returnValue) {
            //check inheriting classes
            for(Reflex::Base_Iterator b = type.Base_Begin(); b != type.Base_End(); ++ b) {
               returnValue = findDataMember(b->ToType(), iName, oError);
               //only stop if we found it or some other error happened
               if(returnValue || parser::kNameDoesNotExist != oError) {
                  break;
               }
            }
         }
         if(returnValue && !returnValue.IsPublic()) {
            returnValue = Reflex::Member();
            oError = parser::kIsNotPublic;
         }
      }
      if(returnValue) {
         oError = parser::kNoError;
      }
      return returnValue;
   }
}
