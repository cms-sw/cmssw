#ifndef PhysicsTools_Utilities_ErrorCodes_h
#define PhysicsTools_Utilities_ErrorCodes_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     ErrorCodes
// 
/**\enum ErrorCodes ErrorCodes.h PhysicsTools/Utilities/interface/ErrorCodes.h

 Description: enum containing the various ways data/function member lookups can fail

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Aug 13 10:37:44 EDT 2008
// $Id: ErrorCodes.h,v 1.1 2008/08/13 19:38:18 chrjones Exp $
//

// system include files

// user include files

// forward declarations

namespace reco {
   namespace parser {
      enum ErrorCodes {
         kNoError = 0,
         kNameDoesNotExist,
         kIsNotPublic,
         kIsStatic,
         kIsNotConst,
         kIsFunctionAddedByROOT,
         kIsConstructor,
         kIsDestructor,
         kIsOperator,
         kWrongNumberOfArguments,
         kWrongArgumentType,
         kOverloaded
      };
   }
}
#endif
