#ifndef DataFormats_Common_FillViewHelperVector_h
#define DataFormats_Common_FillViewHelperVector_h
// -*- C++ -*-
//
// Package:     DataFormats/Common
//
/**\typedef FillViewHelperVector FillViewHelperVector.h "FillViewHelperVector.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 21 Nov 2014 20:12:50 GMT
//

// system include files
#include <vector>
#include <utility>

// user include files
#include "DataFormats/Provenance/interface/ProductID.h"

// forward declarations

namespace edm {
  typedef std::vector<std::pair<edm::ProductID,unsigned long> > FillViewHelperVector;
}
#endif
