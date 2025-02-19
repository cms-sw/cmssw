// ----------------------------------------------------------------------
//
// ELadministrator.cc
//
// Methods of ELadministrator.
//
// History:
//
// 7/2/98  mf   Created
// 7/6/98  mf   Added ELadministratorX knowledge
// 6/16/99 jvr  Attaches a destination when an error is logged and
//              no destinations are attached          $$ JV:1
// 6/18/99 jvr/mf       Constructor for ELadministratorX uses explcitly
//              constructed ELseverityLevels in its init list, rather
//              than objectslike ELabort, which may not yet have
//              been fully constructed.  Repairs a problem with ELcout.
// 6/23/99  mf  Made emptyContextSUpplier static to this comp unit.
// 12/20/99 mf  Added virtual destructor to ELemptyContextSupplier.
// 2/29/00  mf  swapContextSupplier.
// 4/05/00  mf  swapProcess.
// 5/03/00  mf  When aborting, changed exit() to abort() so that dump
//              will reflect true condition without unwinding things.
// 6/6/00  web  Consolidated ELadministrator/X.
// 6/7/00  web  Reflect consolidation of ELdestination/X.
// 6/12/00 web  Attach cerr, rather than cout, in case of no previously-
//              attached destination; using -> USING.
// 3/6/00  mf   Attach taking name to id the destination, getELdestControl()
// 3/14/01 mf   exitThreshold
// 1/10/06 mf   finish()
//
// ---- CMS version
//
// 12/12/05 mf  change exit() to throw edm::exception
//
//-----------------------------------------------------------------------

// Directory of methods:
//----------------------

// ELadministrator::setProcess( const ELstring & process )
// ELadministrator::swapProcess( const ELstring & process )
// ELadministrator::setContextSupplier( const ELcontextSupplier & supplier )
// ELadministrator::getContextSupplier() const
// ELadministrator::swapContextSupplier( ELcontextSupplier & supplier )
// ELadministrator::setAbortThreshold( const ELseverityLevel & sev )
// ELadministrator::setExitThreshold( const ELseverityLevel & sev )
// ELadministrator::attach( const ELdestination & sink )
// ELadministrator::attach( const ELdestination & sink, const ELstring & name )
// ELadministrator::getELdestControl ( const ELstring & name,
//                                      ELdestControl & theDestControl )
// ELadministrator::checkSeverity()
// ELadministrator::severityCount( const ELseverityLevel & sev ) const
// ELadministrator::severityCount( const ELseverityLevel & from,
//                                  const ELseverityLevel & to    ) const
// ELadministrator::resetSeverityCount( const ELseverityLevel & sev )
// ELadministrator::resetSeverityCount( const ELseverityLevel & from,
//                                      const ELseverityLevel & to    )
// ELadministrator::resetSeverityCount()
// ELadministrator::setThresholds( const ELseverityLevel & sev )
// ELadministrator::setLimits( const ELstring & id, int limit )
// ELadministrator::setLimits( const ELseverityLevel & sev, int limit )
// ELadministrator::setIntervals( const ELstring & id, int interval )
// ELadministrator::setIntervals( const ELseverityLevel & sev, int interval )
// ELadministrator::setTimespans( const ELstring & id, int seconds )
// ELadministrator::setTimespans( const ELseverityLevel & sev, int seconds )
// ELadministrator::wipe()
// ELadministrator::finish()
//
// ELadministrator::process() const
// ELadministrator::context() const
// ELadministrator::abortThreshold() const
// ELadministrator::exitThreshold() const
// ELadministrator::sinks()
// ELadministrator::highSeverity() const
// ELadministrator::severityCounts( const int lev ) const
// ELadministrator::finishMsg()
// ELadministrator::clearMsg()
//
// class ELemptyContextSupplier : public ELcontextSupplier
//
// ----------------------------------------------------------------------


#include "FWCore/MessageService/interface/ELadministrator.h"
#include "FWCore/MessageService/interface/ELdestination.h"
#include "FWCore/MessageService/interface/ELcontextSupplier.h"
#include "FWCore/MessageService/interface/ELoutput.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>
#include <sstream>
#include <list>
using std::cerr;


namespace edm {
namespace service {


// Possible Traces:
// #define ELadministratorCONSTRUCTOR_TRACE
// #define ELadTRACE_FINISH


// ----------------------------------------------------------------------
// ELadministrator functionality:
// ----------------------------------------------------------------------

void ELadministrator::setProcess( const ELstring & process )  {

  process_ = process;
  #if 0
    std::cerr << "Administrator process set to \"" << process << "\"\n";
  #endif

}  // setProcess()


ELstring ELadministrator::swapProcess( const ELstring & process )  {

  ELstring temp = process_;
  process_ = process;
  return temp;

} // swapProcess()


void ELadministrator::setContextSupplier( const ELcontextSupplier & supplier )  {

  context_.reset(supplier.clone());

}  // setContextSupplier()


const ELcontextSupplier & ELadministrator::getContextSupplier() const  {

  return *(context_);

}  // getContextSupplier()

ELcontextSupplier & ELadministrator::swapContextSupplier
                                        (ELcontextSupplier & cs)  {
  ELcontextSupplier & save = *(context_);
  context_.reset(&cs);
  return save;
} // swapContextSupplier


void ELadministrator::setAbortThreshold( const ELseverityLevel & sev )  {

  abortThreshold_ = sev;

}  // setAbortThreshold()

void ELadministrator::setExitThreshold( const ELseverityLevel & sev )  {

  exitThreshold_ = sev;

}  // setExitThreshold()

ELdestControl ELadministrator::attach( const ELdestination & sink )  {

  boost::shared_ptr<ELdestination> dest(sink.clone());
  sinks().push_back( dest );
  return ELdestControl( dest );

}  // attach()

ELdestControl ELadministrator::attach(  const ELdestination & sink,
                                        const ELstring & name )     {
  boost::shared_ptr<ELdestination> dest(sink.clone());
  attachedDestinations[name] = dest;
  sinks().push_back( dest );
  return ELdestControl( dest );
} // attach()


bool ELadministrator::getELdestControl ( const ELstring & name,
                                         ELdestControl & theDestControl ) {

  if ( attachedDestinations.find(name) != attachedDestinations.end() ) {
    theDestControl = ELdestControl ( attachedDestinations[name] );
    return true;
  } else {
    return false;
  }

} // getDestControl

ELseverityLevel  ELadministrator::checkSeverity()  {

  const ELseverityLevel  retval( highSeverity_ );
  highSeverity_ = ELzeroSeverity;
  return retval;

}  // checkSeverity()


int ELadministrator::severityCount( const ELseverityLevel & sev ) const  {

  return severityCounts_[sev.getLevel()];

}  // severityCount()


int ELadministrator::severityCount(
  const ELseverityLevel & from,
  const ELseverityLevel & to
)  const  {

  int k = from.getLevel();
  int sum = severityCounts_[k];

  while ( ++k != to.getLevel() )
    sum += severityCounts_[k];

  return  sum;

}  // severityCount()


void ELadministrator::resetSeverityCount( const ELseverityLevel & sev )  {

  severityCounts_[sev.getLevel()] = 0;

}  // resetSeverityCount()


void ELadministrator::resetSeverityCount( const ELseverityLevel & from,
                                          const ELseverityLevel & to   )  {

  for ( int k = from.getLevel();  k <= to.getLevel();  ++k )
    severityCounts_[k] = 0;

}  // resetSeverityCount()


void ELadministrator::resetSeverityCount()  {

  resetSeverityCount( ELzeroSeverity, ELhighestSeverity );

}  // resetSeverityCount()


// ----------------------------------------------------------------------
// Accessors:
// ----------------------------------------------------------------------

const ELstring & ELadministrator::process() const  { return process_; }


ELcontextSupplier & ELadministrator::context() const  { return *context_; }


const ELseverityLevel & ELadministrator::abortThreshold() const  {
  return abortThreshold_;
}

const ELseverityLevel & ELadministrator::exitThreshold() const  {
  return exitThreshold_;
}

std::list<boost::shared_ptr<ELdestination> > & ELadministrator::sinks()  { return sinks_; }


const ELseverityLevel & ELadministrator::highSeverity() const  {
  return highSeverity_;
}


int ELadministrator::severityCounts( const int lev ) const  {
  return severityCounts_[lev];
}


// ----------------------------------------------------------------------
// Message handling:
// ----------------------------------------------------------------------

static inline void msgexit(int s) {
  std::ostringstream os;
  os << "msgexit - MessageLogger requested to exit with status " << s;
  edm::Exception e(edm::errors::LogicError, os.str());
  throw e;
}

static inline void msgabort() {
  std::ostringstream os;
  os << "msgabort - MessageLogger requested to abort";
  edm::Exception e(edm::errors::LogicError, os.str());
  throw e;
}

static inline void possiblyAbortOrExit (int s, int a, int e) {
  if (s < a && s < e) return;
  if (a < e) {
    if ( s < e ) msgabort();
    msgexit(s);
  } else {
    if ( s < a ) msgexit(s);
    msgabort();
  }
}

void ELadministrator::finishMsg()  {

  if ( ! msgIsActive )
    return;

  int lev = msg.xid().severity.getLevel();
  ++ severityCounts_[lev];
  if ( lev > highSeverity_.getLevel() )
    highSeverity_ = msg.xid().severity;

  #ifdef ELadTRACE_FINISH
    std::cerr << "=:=:=: finshMsg() - lev = " << lev << "\n";
  #endif

  context_->editErrorObj( msg );

  #ifdef ELadTRACE_FINISH
    std::cerr << "=:=:=: finshMsg() returns from editErrorObj( msg ) \n";
  #endif

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  bool mrt;
  #ifdef ELadTRACE_FINISH
    int destCounter = 0;
  #endif
  if ( sinks().begin() == sinks().end() )  {                   // $$ JV:1
    std::cerr << "\nERROR LOGGED WITHOUT DESTINATION!\n";
    std::cerr << "Attaching destination \"cerr\" to ELadministrator by default\n\n";
    boost::shared_ptr<ELdestination> dest(new ELoutput(cerr));
    this->sinks().push_back(dest);
  }
  for ( d = sinks().begin();  d != sinks().end();  ++d )  {
    #ifdef ELadTRACE_FINISH
      std::cerr << "  =:=:=: log(msg) for a destination number "
           << ++destCounter << " called ... \n";
    #endif
    mrt = (*d)->log( msg );
    #ifdef ELadTRACE_FINISH
      std::cerr << "  =:=:=: log(msg) for a destination returned " << mrt << "\n";
    #endif
    if ( mrt )
      msg.setReactedTo(true);
  }

  msgIsActive = false;

  possiblyAbortOrExit ( lev,
                        abortThreshold().getLevel(),
                        exitThreshold().getLevel()   );         // $$ mf 3/17/04

}  // finishMsg()


void ELadministrator::clearMsg()  {

  msgIsActive = false;
  msg.clear();

}  // clearMsg()


// ----------------------------------------------------------------------
// The following do the indicated action to all attached destinations:
// ----------------------------------------------------------------------

void ELadministrator::setThresholds( const ELseverityLevel & sev )  {

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = sinks().begin();  d != sinks().end();  ++d )
    (*d)->threshold = sev;

}  // setThresholds()


void ELadministrator::setLimits( const ELstring & id, int limit )  {

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = sinks().begin();  d != sinks().end();  ++d )
    (*d)->limits.setLimit( id, limit );

}  // setLimits()


void ELadministrator::setIntervals
			( const ELseverityLevel & sev, int interval )  {

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = sinks().begin();  d != sinks().end();  ++d )
    (*d)->limits.setInterval( sev, interval );

}  // setIntervals()

void ELadministrator::setIntervals( const ELstring & id, int interval )  {

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = sinks().begin();  d != sinks().end();  ++d )
    (*d)->limits.setInterval( id, interval );

}  // setIntervals()


void ELadministrator::setLimits( const ELseverityLevel & sev, int limit )  {

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = sinks().begin();  d != sinks().end();  ++d )
    (*d)->limits.setLimit( sev, limit );

}  // setLimits()


void ELadministrator::setTimespans( const ELstring & id, int seconds )  {

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = sinks().begin();  d != sinks().end();  ++d )
    (*d)->limits.setTimespan( id, seconds );

}  // setTimespans()


void ELadministrator::setTimespans( const ELseverityLevel & sev, int seconds )  {

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = sinks().begin();  d != sinks().end();  ++d )
    (*d)->limits.setTimespan( sev, seconds );

}  // setTimespans()


void ELadministrator::wipe()  {

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = sinks().begin();  d != sinks().end();  ++d )
    (*d)->limits.wipe();

}  // wipe()

void ELadministrator::finish()  {

  std::list<boost::shared_ptr<ELdestination> >::iterator d;
  for ( d = sinks().begin();  d != sinks().end();  ++d )
    (*d)->finish();

}  // wipe()


// ----------------------------------------------------------------------
// ELemptyContextSupplier (dummy default ELcontextSupplier):
// ----------------------------------------------------------------------

class ELemptyContextSupplier : public ELcontextSupplier  {

public:

  virtual
  ELemptyContextSupplier *
  clone() const  {
    return new ELemptyContextSupplier( *this );
  }
  virtual ELstring context()        const  {  return ""; }
  virtual ELstring summaryContext() const  {  return ""; }
  virtual ELstring fullContext()    const  {  return ""; }

  virtual ~ELemptyContextSupplier()  {}

};  // ELemptyContextSupplier


static ELemptyContextSupplier  emptyContext;


// ----------------------------------------------------------------------
// The Destructable Singleton pattern
// (see "To Kill a Singleton," Vlissides, C++ Report):
// ----------------------------------------------------------------------


ELadministrator * ELadministrator::instance_ = 0;


ELadministrator * ELadministrator::instance()  {

  static ELadminDestroyer destroyer_;
  // This deviates from Vlissides' pattern where destroyer_ was a static
  // instance in the ELadministrator class.  This construct should be
  // equivalent, but the original did not call the destructor under KCC.

  if ( !instance_ )  {
    instance_ = new ELadministrator;
    destroyer_.setELadmin( instance_ );
  }
  return instance_;

}  // instance()


ELadministrator::ELadministrator()
: process_       ( ""                                                        )
, context_       ( emptyContext.clone()                                      )
, abortThreshold_( ELseverityLevel (ELseverityLevel::ELsev_abort)            )
, exitThreshold_ ( ELseverityLevel (ELseverityLevel::ELsev_highestSeverity   )            )
, sinks_         (                                                           )
, highSeverity_  ( ELseverityLevel (ELseverityLevel::ELsev_zeroSeverity)     )
, msg            ( ELseverityLevel (ELseverityLevel::ELsev_unspecified), ""  )
, msgIsActive    ( false                                                     )
{

  #ifdef ELadministratorCONSTRUCTOR_TRACE
    std::cerr << "ELadminstrator constructor\n";
  #endif

  for ( int lev = 0;  lev < ELseverityLevel::nLevels;  ++lev )
    severityCounts_[lev] = 0;

}  // ELadministrator()


ELadminDestroyer::ELadminDestroyer( ELadministrator * ad )  : admin_( ad )  {}


ELadminDestroyer::~ELadminDestroyer()  {

  #ifdef ELadministratorCONSTRUCTOR_TRACE
    std::cerr << "~ELadminDestroyer: Deleting admin_\n";
  #endif

  delete admin_;

}  // ~ELadminDestroyer()


void ELadminDestroyer::setELadmin( ELadministrator * ad )  { admin_ = ad; }


//-*****************************
// The ELadminstrator destructor
//-*****************************

ELadministrator::~ELadministrator()  {

  #ifdef ELadministratorCONSTRUCTOR_TRACE
    std::cerr << "ELadministrator Destructor\n";
  #endif

  finishMsg();

  sinks().clear();

}  // ~ELadministrator()



// ----------------------------------------------------------------------


} // end of namespace service  
} // end of namespace edm  
