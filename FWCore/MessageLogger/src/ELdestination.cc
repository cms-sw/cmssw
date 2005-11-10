// ----------------------------------------------------------------------
//
// ELdestination.cc
//
// History:
//
// 7/5/98 	mf	Created
// 6/16/99      jvr     Allow suppress/include options on destinations
// 7/2/99       jvr     Added separate/attachTime, Epilogue, and Serial
//			options
// 6/7/00	web	Consolidate ELdestination/X; add filterModule()
// 8/22/00	web	Fix omitted .getSymbol() call
// 10/4/00	mf	excludeModule()
// 1/15/01 	mf   	setLineLength()
// 2/13/01	mf	fix written by pc to accomodate NT problem with 
//			static init { $001$ }.  Corresponding fix is in 
//			.h file.
// 3/13/01	mf	statisticsMap()
// 4/05/01	mf	multi-module filtering
// 4/12/01	mf	repair multi-module filtering
// 6/23/03 	mf  	changeFile(), flush() 
//
// ----------------------------------------------------------------------


#include <iostream>

#include "FWCore/MessageLogger/interface/ELdestination.h"
#include "FWCore/MessageLogger/interface/ELdestControl.h"
#include "FWCore/MessageLogger/interface/ELadministrator.h"

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"
#include "FWCore/MessageLogger/interface/ELlist.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"


// Possible Traces:
// #define ELdestinationCONSTRUCTOR_TRACE


namespace edm {       

					 // Fix $001 2/13/01 mf
#ifdef DEFECT_NO_STATIC_CONST_INIT
  const int ELdestination::defaultLineLength = 80;
#endif

ELdestination::ELdestination()
: threshold     ( ELzeroSeverity    )
, traceThreshold( ELhighestSeverity )
, limits        (                   )
, preamble      ( "%ERLOG"          )
, newline       ( "\n"              )
, indent        ( "        "        )
, lineLength    ( defaultLineLength )
, ignoreMostModules (false)
, respondToThese()
, respondToMostModules (false)
, ignoreThese()
, xxxxInt       ( 0                 )
{

  #ifdef ELdestinationCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELdestination\n";
  #endif

}  // ELdestination()


ELdestination::~ELdestination()  {

  #ifdef ELdestinationCONSTRUCTOR_TRACE
    std::cerr << "Destructor for ELdestination\n";
  #endif

}  // ~ELdestination()


// ----------------------------------------------------------------------
// Methods invoked by the ELadministrator:
// ----------------------------------------------------------------------

bool ELdestination::log( const ErrorObj & msg )  { return false; }


// ----------------------------------------------------------------------
// Methods invoked through the ELdestControl handle:
// ----------------------------------------------------------------------

// Each of these must be overridden by any destination for which they make
// sense.   In this base class, where they are all no-ops, the methods which
// generate data to a destination, stream or stream will warn at that place,
// and all the no-op methods will issue an ELwarning2 at their own destination.


static ELstring noSummarizationMsg = "No summarization()";
static ELstring noSummaryMsg = "No summary()";
static ELstring noClearSummaryMsg = "No clearSummary()";
static ELstring hereMsg = "available via this destination";
static ELstring noosMsg = "No ostream";
static ELstring notELoutputMsg = "This destination is not an ELoutput";

void ELdestination::clearSummary()  {

  ErrorObj msg( ELwarning2, noClearSummaryMsg );
  msg << hereMsg;
  log( msg );

}  // clearSummary()


void ELdestination::wipe()  { limits.wipe(); }


void ELdestination::zero()  { limits.zero(); }

void ELdestination::respondToModule( ELstring const & moduleName )  { 
  if (moduleName=="*") {
    ignoreMostModules = false;
    respondToMostModules = true;
    ignoreThese.clear();
    respondToThese.clear();
  } else {
    respondToThese.insert(moduleName);
    ignoreThese.erase(moduleName);
  }
}

void ELdestination::ignoreModule( ELstring const & moduleName )  { 
  if (moduleName=="*") {
    respondToMostModules = false;
    ignoreMostModules = true;
    respondToThese.clear();
    ignoreThese.clear();
  } else {
    ignoreThese.insert(moduleName);
    respondToThese.erase(moduleName);
  }
}

void ELdestination::filterModule( ELstring const & moduleName )  { 
  ignoreModule("*"); 
  respondToModule(moduleName); 
}

void ELdestination::excludeModule( ELstring const & moduleName )  { 
  respondToModule("*"); 
  ignoreModule(moduleName); 
}

void ELdestination::summary( ELdestControl & dest, const ELstring & title )  {

  ErrorObj msg( ELwarning2, noSummaryMsg );
  msg << noSummaryMsg << " " << hereMsg << dest.getNewline() << title;
  dest.log( msg );

}  // summary()


void ELdestination::summary( std::ostream & os, const ELstring & title )  {

  os << "%ERLOG" << ELwarning2.getSymbol() << " "
       << noSummaryMsg << " " << hereMsg << std::endl
     << title << std::endl;

}  // summary()


void ELdestination::summary( ELstring & s, const ELstring & title )  {

  s = ELstring("%ERLOG") + ELwarning2.getSymbol() + " "
      + noSummaryMsg + " " + hereMsg + "\n"
    + title + "\n";

}  // summary()


void ELdestination::setTableLimit( int n )  { limits.setTableLimit( n ); }


void ELdestination::summarization(
  const ELstring & title
, const ELstring & sumLines )  {

  ErrorObj  msg( ELwarning2, noSummarizationMsg );
  msg << hereMsg << newline << title;
  log( msg );

}

std::map<ELextendedID , StatsCount> ELdestination::statisticsMap() const {
  return std::map<ELextendedID , StatsCount> ();
}

void ELdestination::changeFile (std::ostream & os) {
  ErrorObj  msg( ELwarning2, noosMsg );
  msg << notELoutputMsg;
  log( msg );
}

void ELdestination::changeFile (const ELstring & filename) {
  ErrorObj  msg( ELwarning2, noosMsg );
  msg << notELoutputMsg << newline << "file requested is" << filename;
  log( msg );
}

void ELdestination::flush () {
  ErrorObj  msg( ELwarning2, noosMsg );
  msg << "cannot flush()";
  log( msg );
}

// ----------------------------------------------------------------------
// Output format options:
// ----------------------------------------------------------------------

void ELdestination::suppressText()  { ; }                      // $$ jvr
void ELdestination::includeText()   { ; }

void ELdestination::suppressModule()  { ; }
void ELdestination::includeModule()   { ; }

void ELdestination::suppressSubroutine()  { ; }
void ELdestination::includeSubroutine()   { ; }

void ELdestination::suppressTime()  { ; }
void ELdestination::includeTime()   { ; }

void ELdestination::suppressContext()  { ; }
void ELdestination::includeContext()   { ; }

void ELdestination::suppressSerial()  { ; }
void ELdestination::includeSerial()   { ; }

void ELdestination::useFullContext()  { ; }
void ELdestination::useContext()      { ; }

void ELdestination::separateTime()  { ; }
void ELdestination::attachTime()    { ; }

void ELdestination::separateEpilogue()  { ; }
void ELdestination::attachEpilogue()    { ; }

void ELdestination::noTerminationSummary()  { ; }

ELstring ELdestination::getNewline() const  { return newline; }

int ELdestination::setLineLength (int len) { 
  int temp=lineLength;
  lineLength = len;
  return temp;
}

int ELdestination::getLineLength () const { return lineLength; }


// ----------------------------------------------------------------------
// Protected helper methods:
// ----------------------------------------------------------------------

bool ELdestination::thisShouldBeIgnored(const ELstring & s) const {
  if (respondToMostModules) {
    return ( ignoreThese.find(s) != ignoreThese.end() );
  } else if (ignoreMostModules) {
    return ( respondToThese.find(s) == respondToThese.end() );
  } else {
  return false;
  }
}

// ----------------------------------------------------------------------
// Maintenance and test functionality:
// ----------------------------------------------------------------------

void ELdestination::xxxxSet( int i )  { xxxxInt = i; }

void ELdestination::xxxxShout()  {
  std::cerr << "XXXX ELdestination: " << xxxxInt << std::endl;
}

} // end of namespace edm  */
