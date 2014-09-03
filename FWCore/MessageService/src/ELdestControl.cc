// ----------------------------------------------------------------------
//
// ELdestControl.cc
//
// History:
//
// 7/5/98 mf    Created
// 6/16/99 jvr  Allow include/suppress options on destinations
// 7/2/99 jvr   Added separate/attachTime, Epilogue, and Serial options
// 7/2/99 jvr   Added separate/attachTime, Epilogue, and Serial options
// 6/7/00 web   Reflect consolidation of ELdestination/X; add
//              filterModule()
// 6/15/00 web  using -> USING
// 10/4/00 mf   excludeModule()
// 3/13/01 mf   statisticsMap()
// 4/04/01 mf   ignoreModule() and respondToModule();
//10/17/01 mf   setTableLimit()
// 3/03/02 mf   conditionalize all forwarding on if (d) so that using a
//              default ELdestControl has no effect on anything.  Needed for
//              good behavior for handle recovery.
//  6/23/03 mf  changeFile(), flush()
//  5/18/06 mf  setInterval
//  6/19/08 mf  summaryForJobReport()
// ----------------------------------------------------------------------


#include "FWCore/MessageService/interface/ELdestControl.h"
#include "FWCore/MessageService/interface/ELdestination.h"

#include <iostream>
using std::cerr;

// Possible Traces:
// #define ELdestinationCONSTRUCTOR_TRACE


namespace edm {
namespace service {


// ----------------------------------------------------------------------


ELdestControl::ELdestControl( std::shared_ptr<ELdestination> dest )
: d ( dest )
{
  #ifdef ELdestinationCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELdestControl\n";
  #endif
}  // ELdestControl()


ELdestControl::ELdestControl( )
: d ( )
{
  #ifdef ELdestinationCONSTRUCTOR_TRACE
    std::cerr << "Default Constructor for ELdestControl\n";
  #endif
}  // ELdestControl()


ELdestControl::~ELdestControl()  {
  #ifdef ELdestinationCONSTRUCTOR_TRACE
    std::cerr << "Destructor for ELdestControl\n";
  #endif
}  // ~ELdestControl()


// ----------------------------------------------------------------------
// Behavior control methods invoked by the framework
// ----------------------------------------------------------------------

ELdestControl & ELdestControl::setThreshold( const ELseverityLevel & sv )  {
  if (d) d->threshold = sv;
  return  * this;
}


ELdestControl & ELdestControl::setTraceThreshold( const ELseverityLevel & sv )  {
  if (d) d->traceThreshold = sv;
  return  * this;
}


ELdestControl & ELdestControl::setLimit( const ELstring & s, int n )  {
  if (d) d->limits.setLimit( s, n );
  return  * this;
}


ELdestControl & ELdestControl::setInterval
				( const ELseverityLevel & sv, int interval )  {
  if (d) d->limits.setInterval( sv, interval );
  return  * this;
}

ELdestControl & ELdestControl::setInterval( const ELstring & s, int interval )  {
  if (d) d->limits.setInterval( s, interval );
  return  * this;
}


ELdestControl & ELdestControl::setLimit( const ELseverityLevel & sv, int n )  {
  if (d) d->limits.setLimit( sv, n );
  return  * this;
}


ELdestControl & ELdestControl::setTimespan( const ELstring & s, int n )  {
  if (d) d->limits.setTimespan( s, n );
  return  * this;
}


ELdestControl & ELdestControl::setTimespan( const ELseverityLevel & sv, int n )  {
  if (d) d->limits.setTimespan( sv, n );
  return  * this;
}


ELdestControl & ELdestControl::setTableLimit( int n )  {
  if (d) d->limits.setTableLimit( n );
  return  * this;
}


void ELdestControl::suppressText()  { if (d) d->suppressText(); }  // $$ jvr
void ELdestControl::includeText()   { if (d) d->includeText();  }

void ELdestControl::suppressModule()  { if (d) d->suppressModule(); }
void ELdestControl::includeModule()   { if (d) d->includeModule();  }

void ELdestControl::suppressSubroutine()  { if (d) d->suppressSubroutine(); }
void ELdestControl::includeSubroutine()   { if (d) d->includeSubroutine();  }

void ELdestControl::suppressTime()  { if (d) d->suppressTime(); }
void ELdestControl::includeTime()   { if (d) d->includeTime();  }

void ELdestControl::suppressContext()  { if (d) d->suppressContext(); }
void ELdestControl::includeContext()   { if (d) d->includeContext();  }

void ELdestControl::suppressSerial()  { if (d) d->suppressSerial(); }
void ELdestControl::includeSerial()   { if (d) d->includeSerial();  }

void ELdestControl::useFullContext()  { if (d) d->useFullContext(); }
void ELdestControl::useContext()      { if (d) d->useContext();  }

void ELdestControl::separateTime()  { if (d) d->separateTime(); }
void ELdestControl::attachTime()    { if (d) d->attachTime();   }

void ELdestControl::separateEpilogue()  { if (d) d->separateEpilogue(); }
void ELdestControl::attachEpilogue()    { if (d) d->attachEpilogue();   }

void ELdestControl::noTerminationSummary()  {if (d) d->noTerminationSummary(); }

ELdestControl & ELdestControl::setPreamble( const ELstring & preamble )  {
  if (d) d->preamble = preamble;
  return  * this;
}

int ELdestControl::setLineLength (int len) {
  if (d) {
    return d->setLineLength(len);
  } else {
    return 0;
  }
}

int ELdestControl::getLineLength () const {
  if (d) {
    return d->getLineLength();
  } else {
    return 0;
  }
}

void ELdestControl::filterModule( ELstring const & moduleName )  {
  if (d) d->filterModule( moduleName );
}

void ELdestControl::excludeModule( ELstring const & moduleName )  {
  if (d) d->excludeModule( moduleName );
}

void ELdestControl::ignoreModule( ELstring const & moduleName )  {
  if (d) d->ignoreModule( moduleName );
}

void ELdestControl::respondToModule( ELstring const & moduleName )  {
  if (d) d->respondToModule( moduleName );
}


ELdestControl & ELdestControl::setNewline( const ELstring & newline )  {
  if (d) d->newline = newline;
  return  * this;
}


// ----------------------------------------------------------------------
// Active methods invoked by the framework, forwarded to the destination:
// ----------------------------------------------------------------------

// *** Active methods invoked by the framework ***

void ELdestControl::summary( ELdestControl & dest, const char * title )  {
  if (d) d->summary( dest, title );
}


void ELdestControl::summary( std::ostream & os, const char * title )  {
  if (d) d->summary( os, title );
}


void ELdestControl::summary( ELstring & s, const char * title )  {
  if (d) d->summary( s, title );
}

void ELdestControl::summary( )  {
  if (d) d->summary( );
}

void ELdestControl::summaryForJobReport( std::map<std::string, double> & sm)  {
  if (d) d->summaryForJobReport(sm);
}


ELdestControl & ELdestControl::clearSummary()  {
  if (d) d->clearSummary();
  return  * this;
}


ELdestControl & ELdestControl::wipe()  {
  if (d) d->wipe();
  return  * this;
}


ELdestControl & ELdestControl::zero()  {
  if (d) d->zero();
  return  * this;
}


bool ELdestControl::log( edm::ErrorObj & msg )  {
  if (d) {
    return d->log( msg );
  } else {
    return false;
  }
}

void ELdestControl::summarization( const ELstring & title
                                 , const ELstring & sumLines
                                 )  {
  if (d) d->summarization ( title, sumLines );
}

ELstring ELdestControl::getNewline() const  {
  if (d) {
    return d->getNewline();
  } else {
    return ELstring();
  }
}

std::map<ELextendedID , StatsCount> ELdestControl::statisticsMap() const {
  if (d) {
    return d->statisticsMap();
  } else {
    return std::map<ELextendedID , StatsCount>();
  }
}

void ELdestControl::changeFile (std::ostream & os) {
  if (d) d->changeFile(os);
}

void ELdestControl::changeFile (const ELstring & filename) {
  if (d) d->changeFile(filename);
}

void ELdestControl::flush () {
  if (d) d->flush();
}


// ----------------------------------------------------------------------


} // end of namespace service
} // end of namespace edm 
