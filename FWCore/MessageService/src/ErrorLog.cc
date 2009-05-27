// ----------------------------------------------------------------------
//
// ErrorLog.cc
//
// Created 7/7/98 mf
// 5/2/99  web  Added non-default constructor.
// 6/16/99 jvr  Attaches a destination when an ErrorObj is logged and no
//              destinations are attached                       $$ jvr
// 6/6/00  web  Reflect consolidation of ELadministrator/X; consolidate
//              ErrorLog/X
// 6/12/00 web  Attach cerr, rather than cout, in case of no previously-
//              attached destination
// 6/14/00 web  Append missing quote on conditional output
// 3/13/01 mf   hexTrigger and related global methods
// 3/13/01 mf   setDiscardThreshold(), and discardThreshold mechanism
// 3/14/01 web  g/setfill(0)/s//setfill('0')/g; move #include <string>
// 3/14/01 web  Insert missing initializers in constructor
// 5/7/01  mf   operator<< (const char[])
// 6/7/01  mf   operator()(ErrorObj&) should have been doing level bookkeeping
//              and abort checking; inserted this code
// 11/15/01 mf  static_cast to unsigned int and long in comparisons in
//              operator<<( ErrorLog & e, unsigned int n) and long, and
//              also rwriting 0xFFFFFFFF as 0xFFFFFFFFL when comparing to a
//              long.  THese cure warnings when -Wall -pedantic are turned on.
// 3/06/01 mf   getELdestControl() forwards to *a
// 12/3/02 mf   discardVerbosityLevel, and items related to operator()(int)
// 6/23/03 mf   moduleName() and subroutineName()
// 3/17/04 mf   spaces after ints.
// 3/17/04 mf   exit threshold
//
// --- CMS
//
// 12/12/05 mf	replace exit() with throw
//
// ----------------------------------------------------------------------


#include "FWCore/MessageService/interface/ErrorLog.h"
#include "FWCore/MessageService/interface/ELadministrator.h"
#include "FWCore/MessageService/interface/ELdestination.h"
#include "FWCore/MessageService/interface/ELoutput.h"
#include "FWCore/MessageService/interface/ELrecv.h"
#include "FWCore/MessageService/interface/ELcontextSupplier.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>
#include <iomanip>

// Possible Traces:
// #define ErrorLogCONSTRUCTOR_TRACE
// #define ErrorLogOUTPUT_TRACE
// #define ErrorLogENDMSG_TRACE
// #define ErrorLogEMIT_TRACE

#ifdef ErrorLog_EMIT_TRACE
  #include <string>
#endif

namespace edm {
namespace service {



// ----------------------------------------------------------------------
// ErrorLog:
// ----------------------------------------------------------------------

ErrorLog::ErrorLog()
: a( ELadministrator::instance() )
, subroutine( "" )
, module( "" )
, hexTrigger(-1)
, spaceAfterInt(false)
, discardThreshold (ELzeroSeverity)
, discarding (false)
, debugVerbosityLevel(0)
, debugSeverityLevel(ELinfo)
, debugMessageId ("DEBUG")
{

  #ifdef ErrorLogCONSTRUCTOR_TRACE
    std::cout << "Constructor for ErrorLog\n";
  #endif

}  // ErrorLog()

ErrorLog::ErrorLog( const ELstring & pkgName )
: a( ELadministrator::instance() )
, subroutine( "" )
, module( pkgName )
, hexTrigger(-1)
, spaceAfterInt(false)
, discardThreshold (ELzeroSeverity)
, discarding (false)
, debugVerbosityLevel(0)
, debugSeverityLevel(ELinfo)
, debugMessageId ("DEBUG")
{

  #ifdef ErrorLogCONSTRUCTOR_TRACE
    std::cout << "Constructor for ErrorLog (with pkgName = " << pkgName << ")\n";
  #endif

}  // ErrorLog()


ErrorLog::~ErrorLog()  {

  #ifdef ErrorLogCONSTRUCTOR_TRACE
    std::cout << "Destructor for ErrorLog\n";
  #endif

}  // ~ErrorLog()

ErrorLog & ErrorLog::operator() (
  const ELseverityLevel & sev
, const ELstring & id )
{

  if ( sev < discardThreshold ) {
    discarding = true;
    return *this;
  }

  discarding = false;

  #ifdef ErrorLogENDMSG_TRACE
    std::cout << "=:=:=: precautionary endmsg ( "
              << sev.getName() << ", " << id << ")\n";
  #endif

  endmsg();  // precautionary

  // -----  form ErrorObj for this new message:
  //
  a->msgIsActive = true;
  a->msg.set          ( sev, id );
  a->msg.setProcess   ( a->process() );
  a->msg.setModule    ( module );
  a->msg.setSubroutine( subroutine );
  a->msg.setReactedTo ( false );

  return  *this;

}  // operator()( sev, id )


void ErrorLog::setSubroutine( const ELstring & subName )  {

  subroutine = subName;

}  // setSubroutine()

static inline void msgexit(int s) {
  std::ostringstream os;
  os << "msgexit - MessageLogger Log requested to exit with status " << s;
  edm::Exception e(edm::errors::LogicError, os.str());
  throw e;
}

static inline void msgabort() {
  std::ostringstream os;
  os << "msgabort - MessageLogger Log requested to abort";
  edm::Exception e(edm::errors::LogicError, os.str());
  throw e;
}

static inline void possiblyAbOrEx (int s, int a, int e) {
  if (s < a && s < e) return;
  if (a < e) {
    if ( s < e ) msgabort();
    msgexit(s);
  } else {
    if ( s < a ) msgexit(s);
    msgabort();
  }
}

ErrorLog & ErrorLog::operator()( edm::ErrorObj & msg )  {

  #ifdef ErrorLogENDMSG_TRACE
    std::cout << "=:=:=: precautionary endmsg called from operator() (msg) \n";
  #endif

  endmsg();  // precautionary

  // -----  will we need to poke/restore info into the message?
  //
  bool updateProcess   ( msg.xid().process   .length() == 0 );
  bool updateModule    ( msg.xid().module    .length() == 0 );
  bool updateSubroutine( msg.xid().subroutine.length() == 0 );

  // -----  poke, if needed:
  //
  if ( updateProcess    )  msg.setProcess   ( a->process() );
  if ( updateModule     )  msg.setModule    ( module );
  if ( updateSubroutine )  msg.setSubroutine( subroutine );

  // severity level statistics keeping:                 // $$ mf 6/7/01
  int lev = msg.xid().severity.getLevel();
  ++ a->severityCounts_[lev];
  if ( lev > a->highSeverity_.getLevel() )
    a->highSeverity_ = msg.xid().severity;

  a->context_->editErrorObj( msg );

  // -----  send the message to each destination:
  //
  if (a->sinks().begin() == a->sinks().end())  {
    std::cerr << "\nERROR LOGGED WITHOUT DESTINATION!\n";
    std::cerr << "Attaching destination \"cerr\" to ELadministrator by default\n"
              << std::endl;
    a->attach(ELoutput(std::cerr));
  }
  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = a->sinks().begin();  d != a->sinks().end();  ++d )
    if (  (*d)->log( msg )  )
      msg.setReactedTo ( true );

  possiblyAbOrEx ( msg.xid().severity.getLevel(),
                   a->abortThreshold().getLevel(),
                   a->exitThreshold().getLevel()   );   // $$ mf 3/17/04

  // -----  restore, if we poked above:
  //
  if ( updateProcess    )  msg.setProcess( "" );
  if ( updateModule     )  msg.setModule( "" );
  if ( updateSubroutine )  msg.setSubroutine( "" );

  return  *this;

}  // operator()( )


void ErrorLog::setModule( const ELstring & modName )  {

  module = modName;

}  // setModule()


void ErrorLog::setPackage( const ELstring & pkgName )  {

  setModule( pkgName );

}  // setPackage()


ErrorLog & ErrorLog::operator() (int nbytes, char * data)  {

  ELrecv ( nbytes, data, module );
  return  *this;

}  // operator() (nbytes, data)

ErrorLog & ErrorLog::operator<<( void (* f)(ErrorLog &) )  {
  #ifdef ErrorLogOUTPUT_TRACE
    std::cout << "=:=:=: ErrorLog output trace:  f at " << std::hex << f
              << std::endl;
  #endif
  if (discarding) return *this;
  f( *this );
  return  *this;

}  // operator<<()


ErrorLog & ErrorLog::emit( const ELstring & s )  {

  #ifdef ErrorLogEMIT_TRACE
    std::cout << " =:=:=: ErrorLog emit trace:  string is: " << s << "\n";
  #endif

  if ( ! a->msgIsActive )
    (*this) ( ELunspecified, "..." );

  a->msg.emit( s );

  #ifdef ErrorLogEMIT_TRACE
    std::cout << " =:=:=: ErrorLog emit trace:  return from a->msg.emit()\n";
  #endif

  return  *this;

}  // emit()


ErrorLog & ErrorLog::endmsg()  {

  #ifdef ErrorLogENDMSG_TRACE
    std::cout << "=:=:=: endmsg () -- msgIsActive = " << a->msgIsActive
              << std::endl;
  #endif

  if ( a->msgIsActive )  {
    #ifdef ErrorLogENDMSG_TRACE
      std::cout << "=:=:=: endmsg () -- finishMsg started\n";
    #endif
    a->finishMsg();
    #ifdef ErrorLogENDMSG_TRACE
      std::cout << "=:=:=: endmsg () -- finishMsg completed\n";
    #endif
      a->clearMsg();
    }
    return  *this;

}  // endmsg()

// ----------------------------------------------------------------------
// Advanced Control Options:
// ----------------------------------------------------------------------

bool ErrorLog::setSpaceAfterInt(bool space) {
  bool temp = spaceAfterInt;
  spaceAfterInt = space;
  return temp;
}

int ErrorLog::setHexTrigger (int trigger) {
  int oldTrigger = hexTrigger;
  hexTrigger = trigger;
  return oldTrigger;
}

ELseverityLevel ErrorLog::setDiscardThreshold (ELseverityLevel sev) {
  ELseverityLevel oldSev = discardThreshold;
  discardThreshold = sev;
  return oldSev;
}

void ErrorLog::setDebugVerbosity (int debugVerbosity) {
  debugVerbosityLevel = debugVerbosity;
}

void ErrorLog::setDebugMessages (ELseverityLevel sev, ELstring id) {
  debugSeverityLevel = sev;
  debugMessageId = id;
}

bool ErrorLog::getELdestControl (const ELstring & name,
                                       ELdestControl & theDestControl) const {
  return a->getELdestControl(name, theDestControl);
}

// ----------------------------------------------------------------------
// Obtaining Information:
// ----------------------------------------------------------------------

ELstring ErrorLog::moduleName() const { return module; }
ELstring ErrorLog::subroutineName() const { return subroutine; }

// ----------------------------------------------------------------------
// Global endmsg:
// ----------------------------------------------------------------------

void endmsg( ErrorLog & log )  { log.endmsg(); }

// ----------------------------------------------------------------------
// operator<< for integer types
// ----------------------------------------------------------------------

ErrorLog & operator<<( ErrorLog & e, int n)  {
  if (e.discarding) return e;
  std::ostringstream  ost;
  ost << n;
  int m = (n<0) ? -n : n;
  if ( (e.hexTrigger >= 0) && (m >= e.hexTrigger) ) {
    ost << " [0x"
        << std::hex << std::setw(8) << std::setfill('0')
        << n << "] ";
  } else {
    if (e.spaceAfterInt) ost << " ";                    // $$mf 3/17/04
  }
  return e.emit( ost.str() );
}

ErrorLog & operator<<( ErrorLog & e, unsigned int n)  {
  if (e.discarding) return e;
  std::ostringstream  ost;
  ost << n;
  if ( (e.hexTrigger >= 0) &&
       (n >= static_cast<unsigned int>(e.hexTrigger)) ) {
    ost << "[0x"
        << std::hex << std::setw(8) << std::setfill('0')
        << n << "] ";
  } else {
    if (e.spaceAfterInt) ost << " ";                    // $$mf 3/17/04
  }
  return e.emit( ost.str() );
}

ErrorLog & operator<<( ErrorLog & e, long n)  {
  if (e.discarding) return e;
  std::ostringstream  ost;
  ost << n;
  long m = (n<0) ? -n : n;
  if ( (e.hexTrigger >= 0) && (m >= e.hexTrigger) ) {
    int width = 8;
    if ( static_cast<unsigned long>(n) > 0xFFFFFFFFL ) width = 16;
    ost << "[0x"
        << std::hex << std::setw(width) << std::setfill('0')
        << n << "] ";
  } else {
    if (e.spaceAfterInt) ost << " ";                    // $$mf 3/17/04
  }
  return  e.emit( ost.str() );
}

ErrorLog & operator<<( ErrorLog & e, unsigned long n)  {
  if (e.discarding) return e;
  std::ostringstream  ost;
  ost << n;
  if ( (e.hexTrigger >= 0) &&
       (n >= static_cast<unsigned long>(e.hexTrigger)) ) {
    int width = 8;
    if ( n > 0xFFFFFFFFL ) width = 16;
    ost << "[0x"
        << std::hex << std::setw(width) << std::setfill('0')
        << n << "] ";
  } else {
    if (e.spaceAfterInt) ost << " ";                    // $$mf 3/17/04
  }
  return  e.emit( ost.str() );
}

ErrorLog & operator<<( ErrorLog & e, short n)  {
  if (e.discarding) return e;
  std::ostringstream  ost;
  ost << n;
  short m = (n<0) ? -n : n;
  if ( (e.hexTrigger >= 0) && (m >= e.hexTrigger) ) {
    ost << "[0x"
        << std::hex << std::setw(4) << std::setfill('0')
        << n << "] ";
  } else {
    if (e.spaceAfterInt) ost << " ";                    // $$mf 3/17/04
  }
  return  e.emit( ost.str() );
}

ErrorLog & operator<<( ErrorLog & e, unsigned short n)  {
  if (e.discarding) return e;
  std::ostringstream  ost;
  ost << n;
  if ( (e.hexTrigger >= 0) && (n >= e.hexTrigger) ) {
    ost << "[0x"
        << std::hex << std::setw(4) << std::setfill('0')
        << n << "] ";
  } else {
    if (e.spaceAfterInt) ost << " ";                    // $$mf 3/17/04
  }
  return  e.emit( ost.str() );
}


// ----------------------------------------------------------------------
// operator<< for const char[]
// ----------------------------------------------------------------------

ErrorLog & operator<<( ErrorLog & e, const char s[] ) {
  // Exactly equivalent to the general template.
  // If this is not provided explicitly, then the template will
  // be instantiated once for each length of string ever used.
  if (e.discarding) return e;
  std::ostringstream  ost;
  ost << s << ' ';
  return  e.emit( ost.str() );
}


// ----------------------------------------------------------------------


} // end of namespace service  
} // end of namespace edm  
