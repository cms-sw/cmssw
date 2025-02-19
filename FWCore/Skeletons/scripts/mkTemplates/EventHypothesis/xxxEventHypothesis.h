#ifndef xxxEventHypothesis_h
#define xxxEventHypothesis_h
// -*- C++ -*-
//// -*- C++ -*-
//
// Package:    xxxEventHypothesis
// Class:      xxxEventHypothesis
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class xxxEventHypothesis xxxEventHypothesis.cc skelsubys/xxxEventHypothesis/interface/xxxEventHypothesis.h
//!\brief YOUR COMMENTS GO HERE
//!
//!
//! A long description of the event hypothesis class should go here.
//! 
//!
//-------------------------------------------------------------------------------------
//
//
// Original Author:  John Doe
//         Created:  day-mon-xx
// RCS(Id)
//

#include "DataFormats/PatCandidates/interface/HardEventHypothesis.h"


namespace pat {

  class xxxEventHypothesis : public HardEventHypothesis {
  public:

    // This will return static event-wide definitions of the candidate roles.
    static const int N_ROLES = 0;
    static const char *   candidateRoles[N_ROLES];
    static const bool     isVector[N_ROLES];
    virtual int           getNCandidateRoles () const { return N_ROLES; }
    virtual const char *  getCandidateRole  (int i = 0) const;
    virtual bool          getIsVector( int i = 0 ) const;
    virtual int           getSize( int i = 0 ) const; 


    xxxEventHypothesis() {}
    virtual ~xxxEventHypothesis() {}
    
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
