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
// $Id: candidateMethods.cc,v 1.1 2005/10/24 12:59:00 llista Exp $
//

// system include files

// user include files
#include "PhysicsTools/CandAlgos/src/candidateMethods.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"

using namespace reco;

const CandidateMethods& 
reco::candidateMethods() {
  static CandidateMethods s_methods;
  if( ! s_methods.size() ) {
    s_methods[ "momentum" ] = &Candidate::momentum;
    s_methods[ "energy" ] = &Candidate::energy;
    s_methods[ "et" ] = &Candidate::et;
    s_methods[ "mass" ] = &Candidate::mass;
    s_methods[ "massSqr" ] = &Candidate::massSqr;
    s_methods[ "mt" ] = &Candidate::mt;
    s_methods[ "mtSqr" ] = &Candidate::mtSqr;
    s_methods[ "px" ] = &Candidate::px;
    s_methods[ "py" ] = &Candidate::py;
    s_methods[ "pz" ] = &Candidate::pz;
    s_methods[ "pt" ] = &Candidate::pt;
    s_methods[ "phi" ] = &Candidate::phi;
    s_methods[ "theta" ] = &Candidate::theta;
    s_methods[ "x" ] = &Candidate::x;
    s_methods[ "y" ] = &Candidate::y;
    s_methods[ "z" ] = &Candidate::z;
  }
  return s_methods;
}
