#ifndef MessageLogger_ELextendedID_h
#define MessageLogger_ELextendedID_h

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

#include "FWCore/MessageLogger/interface/ELseverityLevel.h"

#include <string>

namespace edm {

  // ----------------------------------------------------------------------
  // ELextendedID:
  // ----------------------------------------------------------------------

  class ELextendedID {
  public:
    // -----  Publicly accessible data members:
    //
    std::string id;
    ELseverityLevel severity;
    std::string module;
    std::string subroutine;

    // -----  Comparator:
    //
    bool operator<(const ELextendedID& xid) const;

    // -----  (Re)initializer:
    //
    void clear();

  };  // ELextendedID

  // ----------------------------------------------------------------------

}  // end of namespace edm

#endif  // MessageLogger_ELextendedID_h
