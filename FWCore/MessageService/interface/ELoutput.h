#ifndef FWCore_MessageService_ELoutput_h
#define FWCore_MessageService_ELoutput_h


// ----------------------------------------------------------------------
//
// ELoutput	is a subclass of ELdestination representing the standard
//		provided destination.
//
// 7/8/98 mf	Created file.
// 6/17/99 jvr	Made output format options available for ELdestControl only
// 7/2/99 jvr	Added separate/attachTime, Epilogue, and Serial options
// 2/22/00 mf	Changed myDetX to myOutputX (to avoid future puzzlement!)
//		and added ELoutput(ox) to cacilitate inherited classes.
// 6/7/00 web	Consolidated ELoutput/X; add filterModule()
// 6/14/00 web	Declare classes before granting friendship; remove using
// 10/4/00 mf	add excludeModule()
//  4/4/01 mf 	Removed moduleOfInterest and moduleToExclude, in favor
//		of using base class method.
//  6/23/03 mf  changeFile(), flush() 
//  6/11/07 mf  changed default for emitAtStart to false  
//
// ----------------------------------------------------------------------

#include "FWCore/MessageService/interface/ELdestination.h"

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"

#include <memory>

namespace edm {       


// ----------------------------------------------------------------------
// prerequisite classes:
// ----------------------------------------------------------------------

class ErrorObj;
namespace service {       


// ----------------------------------------------------------------------
// ELoutput:
// ----------------------------------------------------------------------

class ELoutput : public ELdestination  {

public:

  // ---  Birth/death:
  //
  ELoutput();
  ELoutput( std::ostream & os, bool emitAtStart = false );	// 6/11/07 mf
  ELoutput( const ELstring & fileName, bool emitAtStart = false );
  ELoutput( const ELoutput & orig );
  ~ELoutput() override;

  // ---  Methods invoked by the ELadministrator:
  //
public:
  bool log( const edm::ErrorObj & msg ) override;

protected:
    // trivial clearSummary(), wipe(), zero() from base class
    // trivial three summary(..) from base class

protected:
  // ---  Internal Methods -- Users should not invoke these:
  //
protected:
  void emitToken( const ELstring & s, bool nl=false ) ;

  void suppressTime() override;
  void includeTime() override;
  void suppressModule()override;
  void includeModule() override;
  void suppressSubroutine() override;
  void includeSubroutine() override;
  void suppressText() override;
  void includeText() override;
  void suppressContext() override;
  void includeContext() override;
  void suppressSerial() override;
  void includeSerial() override;
  void useFullContext() override;
  void useContext() override;
  void separateTime() override;
  void attachTime() override;
  void separateEpilogue() override;
  void attachEpilogue() override;

  void changeFile (std::ostream & os) override;
  void changeFile (const ELstring & filename) override;
  void flush() override;


protected:
  // --- member data:
  //
  std::shared_ptr<std::ostream> os;
  int                             charsOnLine;
  edm::ELextendedID               xid;

  bool wantTimestamp
  ,    wantModule
  ,    wantSubroutine
  ,    wantText
  ,    wantSomeContext
  ,    wantSerial
  ,    wantFullContext
  ,    wantTimeSeparate
  ,    wantEpilogueSeparate
  ,    preambleMode
  ;

  // --- Verboten method:
  //
  ELoutput & operator=( const ELoutput & orig ) = delete;

};  // ELoutput


// ----------------------------------------------------------------------


}        // end of namespace service
}        // end of namespace edm


#endif // FWCore_MessageService_ELoutput_h
