#ifndef ELEXTENDEDID_H
#define ELEXTENDEDID_H


// ----------------------------------------------------------------------
//
// ELextendedID.h	is used as a key in maps for both counting toward
//			limits on how many times a destination will react
//			to a type of message, and for statistics.
//
// 07-Jul-1998 mf	Created file.
// 26-Aug-1998 WEB	Updated with ELseverityLevel in place of int.
//
// ----------------------------------------------------------------------


#ifndef ELSTRING_H
  #include "FWCore/MessageLogger/interface/ELstring.h"
#endif

#ifndef ELSEVERITYLEVEL_H
  #include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#endif


namespace edm {       


// ----------------------------------------------------------------------
// ELextendedID:
// ----------------------------------------------------------------------

class ELextendedID  {

public:

  // -----  Publicly accessible data members:
  //
  ELstring        process;
  ELstring        id;
  ELseverityLevel severity;
  ELstring        module;
  ELstring        subroutine;

  // -----  Comparator:
  //
  bool operator<( const ELextendedID & xid ) const;

  // -----  (Re)initializer:
  //
  void clear();

};  // ELextendedID


// ----------------------------------------------------------------------


}        // end of namespace edm


#endif  // ELEXTENDEDID_H
