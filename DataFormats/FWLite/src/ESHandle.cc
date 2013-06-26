// -*- C++ -*-
//
// Package:     FWLite
// Class  :     ESHandle
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Mon Dec 14 15:29:19 CST 2009
// $Id: ESHandle.cc,v 1.2 2010/07/24 14:14:47 wmtan Exp $
//

// system include files

// user include files
#include "DataFormats/FWLite/interface/ESHandle.h"


//
// constants, enums and typedefs
//

static cms::Exception s_exc("ESHandleUnset", "The ESHandle is being accessed without ever being set by a Record");

static void doNotDelete(cms::Exception*) {}

namespace fwlite {
   boost::shared_ptr<cms::Exception> eshandle_not_set_exception() {
      return boost::shared_ptr<cms::Exception>(&s_exc, doNotDelete);
   }   
}
