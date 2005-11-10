#ifndef ELERRORLIST_H
#define ELERRORLIST_H


// ----------------------------------------------------------------------
//
// ELerrorList	is a subclass of ELdestination representing a simple
//		std::list of ErrorObjs's.
//
// 9/21/00 mf	Created file.
// 10/4/00 mf	Add excludeModule()
//  4/4/01 mf   Removed moduleOfInterest and moduleToExclude, in favor
//              of using base class method.
//
// ----------------------------------------------------------------------


#ifndef ELDESTINATION_H
  #include "FWCore/MessageLogger/interface/ELdestination.h"
#endif

#ifndef ERROROBJ_H
  #include "FWCore/MessageLogger/interface/ErrorObj.h"
#endif

#include <list>

namespace edm {       


// ----------------------------------------------------------------------
// prerequisite classes:
// ----------------------------------------------------------------------

class ErrorObj;
class ELdestControl;


// ----------------------------------------------------------------------
// ELerrorList:
// ----------------------------------------------------------------------

class ELerrorList : public ELdestination  {

  friend class ELdestControl;

public:
  // --- PUBLIC member data:  this list is the whole point of the class!
  //
  std::list<ErrorObj> & errorObjs;

public:

  // ---  Birth/death:
  //
  ELerrorList ( std::list<ErrorObj> & errorList );
  ELerrorList ( const ELerrorList & orig );
  virtual ~ELerrorList();

  // ---  Methods invoked by the ELadministrator:
  //
  virtual
  ELerrorList *
  clone() const;
  // Used by attach() to put the destination on the ELadministrators list
                //-| There is a note in Design Notes about semantics
                //-| of copying a destination onto the list:  ofstream
                //-| ownership is passed to the new copy.

  virtual bool log( const ErrorObj & msg );

  // ---  Methods invoked through the ELdestControl handle:
  //
protected:
    // trivial clearSummary(), wipe(), zero() from base class
    // trivial three summary(..) from base class

protected:
  // --- member data:
  //

  // --- Verboten method:
  //
  ELerrorList & operator=( const ELerrorList & orig );

};  // ELerrorList


// ----------------------------------------------------------------------


}        // end of namespace edm


#endif // ELERRORLIST_H
