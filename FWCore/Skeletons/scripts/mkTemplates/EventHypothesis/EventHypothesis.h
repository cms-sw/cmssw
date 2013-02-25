#ifndef __class___h
#define __class___h
// -*- C++ -*-
//
// Package:    __pkgname__
// Class:      __class__
//
/**


*/
//-------------------------------------------------------------------------------------
//\class __class__ __class__.cc __subsys__/__package__/plugins/__class__.h
//\brief YOUR COMMENTS GO HERE
//
//
// A long description of the event hypothesis class should go here.
// 
//
//-------------------------------------------------------------------------------------
//
//
// Original Author:  __author__
//         Created:  __date__
// __rcsid__
//

#include "DataFormats/PatCandidates/interface/HardEventHypothesis.h"


namespace pat {

  class __class__ : public HardEventHypothesis {
  public:

    // This will return static event-wide definitions of the candidate roles.
    static const int N_ROLES = 0;
    static const char *   candidateRoles[N_ROLES];
    static const bool     isVector[N_ROLES];
    virtual int           getNCandidateRoles () const { return N_ROLES; }
    virtual const char *  getCandidateRole  (int i = 0) const;
    virtual bool          getIsVector( int i = 0 ) const;
    virtual int           getSize( int i = 0 ) const; 


    __class__() {}
    virtual ~__class__() {}
    
    // This is where the event-specific interface will go
PutMyEventHypothesisInterfaceHere;
    
  protected:

    // These must be implemented so that this class can be used by the StarterKit to
    // automatically make plots.
    virtual Candidate &                getCandidate       (std::string name, int index = -1);


    // This is where the event-specific data members will go
PutMyEventHypothesisDataMembersHere;
   };

}

#endif
