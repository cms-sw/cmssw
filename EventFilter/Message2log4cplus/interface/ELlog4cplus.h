#ifndef EventFilter_ELlog4cplus_h
#define EventFilter_ELlog4cplus_h 2

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
//
// ----------------------------------------------------------------------

#include <sstream>


#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"
#include "FWCore/MessageService/interface/ELdestination.h"

#include "EventFilter/Utilities/interface/RunBase.h"

namespace xdaq {
  class Application;
}

namespace edm {       


// ----------------------------------------------------------------------
// prerequisite classes:
// ----------------------------------------------------------------------

class ErrorObj;
class ELdestControl;


// ----------------------------------------------------------------------
// ELoutput:
// ----------------------------------------------------------------------

class ELlog4cplus : public service::ELdestination, public evf::RunBase  {

  friend class service::ELdestControl;

public:

  // ---  Birth/death:
  //
  ELlog4cplus();
  ELlog4cplus( const ELlog4cplus & orig );
  virtual ~ELlog4cplus();

  // ---  Methods invoked by the ELadministrator:
  //
public:
  virtual
  ELlog4cplus *
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
    // ELlog4cplus uses the generic ELdestControl handle

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
			     

  // ---  Maintenance and Testing Methods -- Users should not invoke these:
  //
public:
  void xxxxSet( int i );  // Testing only
  void xxxxShout();       // Testing only
  void setAppl(xdaq::Application *a);

protected:
  // --- member data:
  //
  std::ostringstream os_;
  std::ostringstream *  os;
  bool            osIsOwned;
  int             charsOnLine;
  ELextendedID    xid;

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

  // *** Maintenance and Testing Data ***
  int xxxxInt;             // Testing only

  // --- Verboten method:
  //
  ELlog4cplus & operator=( const ELlog4cplus & orig );

 private:

  xdaq::Application *appl_;

};  // ELlog4cplus


// ----------------------------------------------------------------------


}        // end of namespace edm


#endif // ELlog4cplus_h
