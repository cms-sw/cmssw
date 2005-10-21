// -*- C++ -*-
//
// Package:     CandCombiner
// Class  :     candidateMethods
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Aug 11 18:56:23 EDT 2005
// $Id$
//

// system include files

// user include files
#include "PhysicsTools/CandUtils/interface/candidateMethods.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"

using namespace aod;

const CandidateMethods& 
aod::candidateMethods() 
{
  static CandidateMethods s_methods;
  if( ! s_methods.size() ) {
    s_methods[ "mass" ] = &Candidate::mass;
    s_methods[ "pt" ] = &Candidate::pt;
    s_methods[ "momentum" ] = &Candidate::momentum;

  }
  return s_methods;
}
