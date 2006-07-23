#ifndef CANDCOMBINER_CANDIDATEMETHODS_H
#define CANDCOMBINER_CANDIDATEMETHODS_H
// -*- C++ -*-
//
// Package:     CandCombiner
// Class  :     candidateMethods
// 
/**\class candidateMethods candidateMethods.h PhysicsTools/CandCombiner/interface/candidateMethods.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Aug 11 18:56:27 EDT 2005
// $Id: candidateMethods.h,v 1.2 2006/02/21 10:37:28 llista Exp $
//

#include <map>
#include <boost/function.hpp>

namespace reco {
  class Candidate;
}

namespace reco {
  typedef boost::function<double ( reco::Candidate const& )> PCandMethod;
  typedef std::map<std::string, PCandMethod> CandidateMethods;
  const CandidateMethods& candidateMethods() ;
}

#endif /* CANDCOMBINER_CANDIDATEMETHODS_H */
