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
  virtual ~ELoutput();

  // ---  Methods invoked by the ELadministrator:
  //
public:
  virtual bool log( const edm::ErrorObj & msg ) override;

protected:
    // trivial clearSummary(), wipe(), zero() from base class
    // trivial three summary(..) from base class

protected:
  // ---  Internal Methods -- Users should not invoke these:
  //
protected:
  void emitToken( const ELstring & s, bool nl=false ) ;

  virtual void suppressTime() override;
  virtual void includeTime() override;
  virtual void suppressModule()override;
  virtual void includeModule() override;
  virtual void suppressSubroutine() override;
  virtual void includeSubroutine() override;
  virtual void suppressText() override;
  virtual void includeText() override;
  virtual void suppressContext() override;
  virtual void includeContext() override;
  virtual void suppressSerial() override;
  virtual void includeSerial() override;
  virtual void useFullContext() override;
  virtual void useContext() override;
  virtual void separateTime() override;
  virtual void attachTime() override;
  virtual void separateEpilogue() override;
  virtual void attachEpilogue() override;

  virtual void changeFile (std::ostream & os) override;
  virtual void changeFile (const ELstring & filename) override;
  virtual void flush() override;


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
