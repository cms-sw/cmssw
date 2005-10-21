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
// $Id: candidateMethods.h,v 1.1 2005/10/21 13:56:43 llista Exp $
//

#include <map>

namespace aod {
  class Candidate;
}

namespace aod {
  
  typedef double (aod::Candidate::* PCandMethod)() const;
  typedef std::map<std::string, PCandMethod> CandidateMethods;

  const CandidateMethods& candidateMethods() ;
}

#endif /* CANDCOMBINER_CANDIDATEMETHODS_H */
