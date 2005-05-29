#ifndef EVENTSETUP_RECORDGETIMPLEMENTATION_H
#define EVENTSETUP_RECORDGETIMPLEMENTATION_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     recordGetImplementation
// 
/**\class recordGetImplementation recordGetImplementation.h Core/CoreFramework/interface/recordGetImplementation.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 15:29:02 EST 2005
// $Id: recordGetImplementation.h,v 1.2 2005/04/21 15:47:17 chrjones Exp $
//

// system include files

// user include files

// forward declarations

namespace edm {
   namespace eventsetup {
      template < typename RecordT, typename T > 
      void recordGetImplementation( const RecordT& iRecord ,
                                    T const *& iData ,
                                    const char* iName );
   }
}

#include "FWCore/CoreFramework/interface/recordGetImplementation.icc"

#endif /* EVENTSETUP_RECORDGETIMPLEMENTATION_H */
