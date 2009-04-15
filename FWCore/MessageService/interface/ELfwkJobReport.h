#ifndef FWCore_MessageService_ELfwkJobReport_h
#define FWCore_MessageService_ELfwkJobReport_h


// ----------------------------------------------------------------------
//
// ELfwkJobReport  is a subclass of ELdestination formating in a way
//		   that is good for automated scanning.
//
// 1/10/06 mf, de  Created file.
//
// ----------------------------------------------------------------------

#include "FWCore/MessageService/interface/ELdestination.h"

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"

#include "boost/shared_ptr.hpp"

namespace edm {       


// ----------------------------------------------------------------------
// prerequisite classes:
// ----------------------------------------------------------------------

class ErrorObj;
namespace service {       
class ELdestControl;


// ----------------------------------------------------------------------
// ELfwkJobReport:
// ----------------------------------------------------------------------

class ELfwkJobReport : public ELdestination  {

  friend class ELdestControl;

public:

  // ---  Birth/death:
  //
  ELfwkJobReport();
  ELfwkJobReport( std::ostream & os, bool emitAtStart = true );
  ELfwkJobReport( const ELstring & fileName, bool emitAtStart = true );
  ELfwkJobReport( const ELfwkJobReport & orig );
  virtual ~ELfwkJobReport();

  // ---  Methods invoked by the ELadministrator:
  //
public:
  virtual
  ELfwkJobReport *
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

  // ---  Data affected by methods of specific ELdestControl handle:
  //
protected:
    // ELfwkJobReport uses the generic ELdestControl handle

  // ---  Internal Methods -- Users should not invoke these:
  //
protected:
  virtual void emit( const ELstring & s, bool nl=false );

  virtual void suppressTime();        virtual void includeTime();
  virtual void suppressModule();      virtual void includeModule();
  virtual void suppressSubroutine();  virtual void includeSubroutine();
  virtual void suppressText();        virtual void includeText();
  virtual void suppressContext();     virtual void includeContext();
  virtual void suppressSerial();      virtual void includeSerial();
  virtual void useFullContext();      virtual void useContext();
  virtual void separateTime();        virtual void attachTime();
  virtual void separateEpilogue();    virtual void attachEpilogue();

  virtual void summarization ( const ELstring & fullTitle
                             , const ELstring & sumLines );
			     
  virtual void changeFile (std::ostream & os);
  virtual void changeFile (const ELstring & filename);
  virtual void flush(); 				       
  virtual void finish(); 				       


protected:
  // --- member data:
  //
  boost::shared_ptr<std::ostream> os;
  int                             charsOnLine;
  ELextendedID                    xid;

  bool wantTimestamp
  ,    wantModule
  ,    wantSubroutine
  ,    wantText
  ,    wantSomeContext
  ,    wantSerial
  ,    wantFullContext
  ,    wantTimeSeparate
  ,    wantEpilogueSeparate
  ;

  // --- Verboten method:
  //
  ELfwkJobReport & operator=( const ELfwkJobReport & orig );

};  // ELfwkJobReport


// ----------------------------------------------------------------------


}        // end of namespace service
}        // end of namespace edm


#endif // FWCore_MessageService_ELfwkJobReport_h
