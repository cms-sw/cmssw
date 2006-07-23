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
// $Id: candidateMethods.cc,v 1.5 2006/07/20 10:13:40 llista Exp $
//

// system include files
#include <boost/bind.hpp>
#include <functional>

// user include files
#include "PhysicsTools/CandAlgos/src/candidateMethods.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

const CandidateMethods& 
reco::candidateMethods() {
  static CandidateMethods s_methods;
  if( ! s_methods.size() ) {
    s_methods[ "p" ] = boost::bind(&Candidate::p,_1);
    s_methods[ "energy" ] = boost::bind(&Candidate::energy,_1);
    s_methods[ "et" ] = boost::bind(&Candidate::et,_1);
    s_methods[ "mass" ] = boost::bind(&Candidate::mass,_1);
    s_methods[ "massSqr" ] = boost::bind(&Candidate::massSqr,_1);
    s_methods[ "mt" ] = boost::bind(&Candidate::mt,_1);
    s_methods[ "mtSqr" ] = boost::bind(&Candidate::mtSqr,_1);
    s_methods[ "p" ] = boost::bind(&Candidate::p,_1);
    s_methods[ "px" ] = boost::bind(&Candidate::px,_1);
    s_methods[ "py" ] = boost::bind(&Candidate::py,_1);
    s_methods[ "pz" ] = boost::bind(&Candidate::pz,_1);
    s_methods[ "pt" ] = boost::bind(&Candidate::pt,_1);
    s_methods[ "phi" ] = boost::bind(&Candidate::phi,_1);
    s_methods[ "eta" ] = boost::bind(&Candidate::eta,_1);
    s_methods[ "theta" ] = boost::bind(&Candidate::theta,_1);
    s_methods[ "y" ] = boost::bind(&Candidate::y,_1);
    s_methods[ "vX" ] = boost::bind(&Candidate::vx,_1);
    s_methods[ "vy" ] = boost::bind(&Candidate::vy,_1);
    s_methods[ "vz" ] = boost::bind(&Candidate::vz,_1);
    //NOTE: we are using the 'std::plus<double>' with a 0 to convert
    // the int returned from Candidate::charge into a double
    // the compiler should be able to optimize away the addition
/*    s_methods[ "charge" ] = boost::bind( std::plus<double>(),
					 0,
					 boost::bind(&Candidate::charge,_1));*/
    s_methods[ "charge" ] = boost::bind(&Candidate::charge,_1);
  }
  return s_methods;
}
