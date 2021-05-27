#ifndef L1Trigger_CSCTrackFinder_parameters_h
#define L1Trigger_CSCTrackFinder_parameters_h
// -*- C++ -*-
//
// Package:     L1Trigger/CSCTrackFinders
// Class  :     parameters
//
/**\function parameters parameters.h 

 Description: Converts info in L1MuCSCTFConfiguration to a edm::ParameterSet

 Usage:
    
*/
//
// Original Author:  Christopher Jones
//         Created:  Thu, 27 May 2021 20:02:19 GMT
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class L1MuCSCTFConfiguration;

// forward declarations
edm::ParameterSet parameters(L1MuCSCTFConfiguration const&, int sp);

#endif
