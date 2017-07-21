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
// ELadministrator::attach( const ELdestination & sink )
// ELadministrator::attach( const ELdestination & sink, const ELstring & name )
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
// ELadministrator::sinks_
// ELadministrator::highSeverity() const
// ELadministrator::severityCounts( const int lev ) const
// ELadministrator::finishMsg()
// ELadministrator::clearMsg()
//
//
// ----------------------------------------------------------------------


#include "FWCore/MessageService/interface/ELadministrator.h"
#include "FWCore/MessageService/interface/ELdestination.h"
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

  
void ELadministrator::log(edm::ErrorObj & msg) {
  
  
  // severity level statistics keeping:                 // $$ mf 6/7/01
  int lev = msg.xid().severity.getLevel();
  ++ severityCounts_[lev];
  if ( lev > highSeverity_.getLevel() )
    highSeverity_ = msg.xid().severity;
  
  // -----  send the message to each destination:
  //
  if (sinks_.begin() == sinks_.end())  {
    std::cerr << "\nERROR LOGGED WITHOUT DESTINATION!\n";
    std::cerr << "Attaching destination \"cerr\" to ELadministrator by default\n"
    << std::endl;
    attach(std::make_shared<ELoutput>(std::cerr));
  }
  for (auto& sink : sinks_)
    if ( sink->log( msg )  )
      msg.setReactedTo ( true );
  
  return;
  
}


// ----------------------------------------------------------------------
// ELadministrator functionality:
// ----------------------------------------------------------------------

std::shared_ptr<ELdestination> ELadministrator::attach( std::shared_ptr<ELdestination> sink )  {

  sinks_.push_back( sink );
  return sink;
  
}  // attach()

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

const ELseverityLevel & ELadministrator::highSeverity() const  {
  return highSeverity_;
}


int ELadministrator::severityCounts( const int lev ) const  {
  return severityCounts_[lev];
}


// ----------------------------------------------------------------------
// Message handling:
// ----------------------------------------------------------------------


// ----------------------------------------------------------------------
// The following do the indicated action to all attached destinations:
// ----------------------------------------------------------------------

void ELadministrator::setThresholds( const ELseverityLevel & sev )  {

  for (auto& sink : sinks_)
    sink->threshold = sev;

}  // setThresholds()


void ELadministrator::setLimits( const ELstring & id, int limit )  {

  for (auto& sink : sinks_)
    sink->limits.setLimit( id, limit );

}  // setLimits()


void ELadministrator::setIntervals
			( const ELseverityLevel & sev, int interval )  {

  for (auto& sink : sinks_)
    sink->limits.setInterval( sev, interval );

}  // setIntervals()

void ELadministrator::setIntervals( const ELstring & id, int interval )  {

  for (auto& sink : sinks_)
    sink->limits.setInterval( id, interval );

}  // setIntervals()


void ELadministrator::setLimits( const ELseverityLevel & sev, int limit )  {

  for (auto& sink : sinks_)
    sink->limits.setLimit( sev, limit );

}  // setLimits()


void ELadministrator::setTimespans( const ELstring & id, int seconds )  {

  for (auto& sink : sinks_)
    sink->limits.setTimespan( id, seconds );

}  // setTimespans()


void ELadministrator::setTimespans( const ELseverityLevel & sev, int seconds )  {

  for (auto& sink : sinks_)
    sink->limits.setTimespan( sev, seconds );

}  // setTimespans()


void ELadministrator::wipe()  {

  for (auto& sink : sinks_)
    sink->limits.wipe();

}  // wipe()

void ELadministrator::finish()  {

  for (auto& sink : sinks_)
    sink->finish();

}  // wipe()


ELadministrator::ELadministrator()
: sinks_         (                                                           )
, highSeverity_  ( ELseverityLevel (ELseverityLevel::ELsev_zeroSeverity)     )
{

  #ifdef ELadministratorCONSTRUCTOR_TRACE
    std::cerr << "ELadminstrator constructor\n";
  #endif

  for ( int lev = 0;  lev < ELseverityLevel::nLevels;  ++lev )
    severityCounts_[lev] = 0;

}
//-*****************************
// The ELadminstrator destructor
//-*****************************

ELadministrator::~ELadministrator()  {

  #ifdef ELadministratorCONSTRUCTOR_TRACE
    std::cerr << "ELadministrator Destructor\n";
  #endif

  sinks_.clear();

}  // ~ELadministrator()



// ----------------------------------------------------------------------


} // end of namespace service  
} // end of namespace edm  
