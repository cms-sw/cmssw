#ifndef FWCore_MessageService_ELerrorList_h
#define FWCore_MessageService_ELerrorList_h


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


#include "FWCore/MessageService/interface/ELdestination.h"

#include "FWCore/MessageLogger/interface/ErrorObj.h"

#include <list>

namespace edm {       
namespace service {       


// ----------------------------------------------------------------------
// prerequisite classes:
// ----------------------------------------------------------------------

class ELdestControl;


// ----------------------------------------------------------------------
// ELerrorList:
// ----------------------------------------------------------------------

class ELerrorList : public ELdestination  {

  friend class ELdestControl;

public:
  // --- PUBLIC member data:  this list is the whole point of the class!
  //
  std::list<edm::ErrorObj> & errorObjs;

public:

  // ---  Birth/death:
  //
  ELerrorList ( std::list<edm::ErrorObj> & errorList );
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

  virtual bool log( const edm::ErrorObj & msg );

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


}        // end of namespace service
}        // end of namespace edm


#endif // FWCore_MessageService_ELerrorList_h
