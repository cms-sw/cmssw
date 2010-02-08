// ----------------------------------------------------------------------
//
// ELfwkJobReport.cc
//
//
// 1/10/06      mf, de  Created
//
// Changes:
//
//   1 - 3/22/06  mf  - in configure_dest()	
//	Repaired the fact that destination limits for categories
//	were not being effective:
//	a) use values from the destination specific default PSet
//	   rather than the overall default PSet to set these
//	b) when an explicit value has been set - either by overall default or 
//	   by a destination specific default PSet - set that limit or
//	   timespan for that dest_ctrl via a "*" msgId.
//
// ----------------------------------------------------------------------


#include "FWCore/MessageService/interface/ELfwkJobReport.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

// Possible Traces:
// #define ELfwkJobReportCONSTRUCTOR_TRACE
// #define ELfwkJobReportTRACE_LOG
// #define ELfwkJobReport_EMIT_TRACE

#include <iostream>
#include <fstream>

namespace edm {
namespace service {

// ----------------------------------------------------------------------
// Constructors:
// ----------------------------------------------------------------------

ELfwkJobReport::ELfwkJobReport()
: ELdestination       (            )
, os                  ( &std::cerr, do_nothing_deleter() )
, charsOnLine         ( 0          )
, xid                 (            )
, wantTimestamp       ( true       )
, wantModule          ( true       )
, wantSubroutine      ( true       )
, wantText            ( true       )
, wantSomeContext     ( true       )
, wantSerial          ( false      )
, wantFullContext     ( false      )
, wantTimeSeparate    ( false      )
, wantEpilogueSeparate( false      )
{

  #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELfwkJobReport()\n";
  #endif

  // Opening xml tag
  emitToken( "<FrameworkJobReport>\n", true );

}  // ELfwkJobReport()


ELfwkJobReport::ELfwkJobReport( std::ostream & os_ , bool emitAtStart )
: ELdestination       (       )
, os                  ( &os_, do_nothing_deleter() )
, charsOnLine         ( 0     )
, xid                 (       )
, wantTimestamp       ( true  )
, wantModule          ( true  )
, wantSubroutine      ( true  )
, wantText            ( true  )
, wantSomeContext     ( true  )
, wantSerial          ( false )
, wantFullContext     ( false )
, wantTimeSeparate    ( false )
, wantEpilogueSeparate( false )
{

  #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELfwkJobReport( os )\n";
  #endif

  // Opening xml tag
  emitToken( "<FrameworkJobReport>\n\n", true );

}  // ELfwkJobReport()


ELfwkJobReport::ELfwkJobReport( const ELstring & fileName, bool emitAtStart )
: ELdestination       (       )
, os                  ( new std::ofstream( fileName.c_str() , std::ios/*_base*/::app), close_and_delete())
, charsOnLine         ( 0     )
, xid                 (       )
, wantTimestamp       ( true  )
, wantModule          ( true  )
, wantSubroutine      ( true  )
, wantText            ( true  )
, wantSomeContext     ( true  )
, wantSerial          ( false )
, wantFullContext     ( false )
, wantTimeSeparate    ( false )
, wantEpilogueSeparate( false )
{

  #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELfwkJobReport( " << fileName << " )\n";
  #endif

  if ( os && *os )  {
    #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
      std::cerr << "          Testing if os is owned\n";
    #endif
    #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
      std::cerr << "          About to do first emit\n";
    #endif
    // Opening xml tag
    emitToken( "<FrameworkJobReport>\n");
  } else  {
    #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
      std::cerr << "          Deleting os\n";
    #endif
    os.reset(&std::cerr, do_nothing_deleter());
    #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
      std::cerr << "          about to emit to cerr\n";
    #endif
    // Opening xml tag
    emitToken( "<FrameworkJobReport>\n\n" );
  }

  #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELfwkJobReport completed.\n";
  #endif

}  // ELfwkJobReport()


ELfwkJobReport::ELfwkJobReport( const ELfwkJobReport & orig )
: ELdestination       (                           )
, os                  ( orig.os                   )
, charsOnLine         ( orig.charsOnLine          )
, xid                 ( orig.xid                  )
, wantTimestamp       ( orig.wantTimestamp        )
, wantModule          ( orig.wantModule           )
, wantSubroutine      ( orig.wantSubroutine       )
, wantText            ( orig.wantText             )
, wantSomeContext     ( orig.wantSomeContext      )
, wantSerial          ( orig.wantSerial           )
, wantFullContext     ( orig.wantFullContext      )
, wantTimeSeparate    ( orig.wantTimeSeparate     )
, wantEpilogueSeparate( orig.wantEpilogueSeparate )
{

  #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
    std::cerr << "Copy constructor for ELfwkJobReport\n";
  #endif

  // mf 6/15/01 fix of Bug 005
  threshold             = orig.threshold;
  traceThreshold        = orig.traceThreshold;
  limits                = orig.limits;
  preamble              = orig.preamble;
  newline               = orig.newline;
  indent                = orig.indent;
  lineLength            = orig.lineLength;

  ignoreMostModules     = orig.ignoreMostModules;
  respondToThese        = orig.respondToThese;
  respondToMostModules  = orig.respondToMostModules;
  ignoreThese           = orig.ignoreThese;

}  // ELfwkJobReport()


ELfwkJobReport::~ELfwkJobReport()  {

  #ifdef ELfwkJobReportCONSTRUCTOR_TRACE
    std::cerr << "Destructor for ELfwkJobReport\n";
  #endif
}  // ~ELfwkJobReport()


// ----------------------------------------------------------------------
// Methods invoked by the ELadministrator:
// ----------------------------------------------------------------------

ELfwkJobReport *
ELfwkJobReport::clone() const  {

  return new ELfwkJobReport( *this );

} // clone()


bool ELfwkJobReport::log( const edm::ErrorObj & msg )  {

  #ifdef ELfwkJobReportTRACE_LOG
    std::cerr << "    =:=:=: Log to an ELfwkJobReport \n";
  #endif

  xid = msg.xid();      // Save the xid.

  // Change log 1:  React ONLY to category FwkJob
  if (xid.id != "FwkJob") return false;
  
  // See if this message is to be acted upon
  // (this is redundant if we are reacting only to FwkJob)
  // and add it to limits table if it was not already present:
  //
  if ( msg.xid().severity < threshold  )  return false;
  
  if ( (xid.id == "BeginningJob")        ||
       (xid.id == "postBeginJob")        ||
       (xid.id == "preEventProcessing")  ||
       (xid.id == "preModule")           ||
       (xid.id == "postModule")          ||
       (xid.id == "postEventProcessing") ||
       (xid.id == "postEndJob")             ) return false; 
  if ( thisShouldBeIgnored(xid.module) )  return false;
  if ( ! limits.add( msg.xid() )       )  return false;

  #ifdef ELfwkJobReportTRACE_LOG
    std::cerr << "    =:=:=: Limits table work done \n";
  #endif

  // Output the prologue:
  //
  //emitToken( "  <Report>\n" );
  //emitToken( "    <Severity> " );
  //emitToken(xid.severity.getSymbol());
  //emitToken(" </Severity>\n");
  //emitToken( "    <Category> ");
  //emitToken(xid.id);
  //emitToken( " </Category>\n");
  //emitToken( "    <Message> \n");
  
 //  emitToken( msg.idOverflow() ); this is how to get the rest of the category

  #ifdef ELfwkJobReportTRACE_LOG
    std::cerr << "    =:=:=: Prologue done \n";
  #endif

  // Output each item in the message:
  //
  if ( wantText )  {
    ELlist_string::const_iterator it;
    for ( it = msg.items().begin();  it != msg.items().end();  ++it )  {
    #ifdef ELfwkJobReportTRACE_LOG
      std::cerr << "      =:=:=: Item:  " << *it << '\n';
    #endif
      //  emitToken( "      <Item> " );
      emitToken( *it);
      emitToken( "\n" );
      //emitToken( " </Item>\n" );
    }
  }

  // Close the body of the message
  //emitToken("    </Message>\n");
  
  // Provide further identification: Module
  //
  //emitToken("    <Module> ");
  //emitToken( xid.module );
  //emitToken(" </Module>\n");    

  #ifdef ELfwkJobReportTRACE_LOG
    std::cerr << "    =:=:=: Module done \n";
  #endif

  // close report
  //
  //emitToken("  </Report>\n\n");

  #ifdef ELfwkJobReportTRACE_LOG
    std::cerr << "  =:=:=: log(msg) done: \n";
  #endif

  return true;

}  // log()

void ELfwkJobReport::finish()   {
  // closing xml tag
  (*os) << "</FrameworkJobReport>\n";
}

// Remainder are from base class.


// ----------------------------------------------------------------------
// Output methods:
// ----------------------------------------------------------------------

void ELfwkJobReport::emitToken( const ELstring & s, bool nl )  {

  #ifdef ELfwkJobReport_EMIT_TRACE
    std::cerr << "[][][] in emit:  charsOnLine is " << charsOnLine << '\n';
    std::cerr << "[][][] in emit:  s.length() " << s.length() << '\n';
    std::cerr << "[][][] in emit:  lineLength is " << lineLength << '\n';
  #endif

  if (s.length() == 0)  {
    return;
  }

  #ifdef ELfwkJobReport_EMIT_TRACE
    std::cerr << "[][][] in emit: about to << s to *os: " << s << " \n";
  #endif

  (*os) << s;

  #ifdef ELfwkJobReport_EMIT_TRACE
    std::cerr << "[][][] in emit: completed \n";
  #endif

}  // emitToken()


// ----------------------------------------------------------------------
// Methods controlling message formatting:
// ----------------------------------------------------------------------

void ELfwkJobReport::includeTime()   { wantTimestamp = true;  }
void ELfwkJobReport::suppressTime()  { wantTimestamp = false; }

void ELfwkJobReport::includeModule()   { wantModule = true;  }
void ELfwkJobReport::suppressModule()  { wantModule = false; }

void ELfwkJobReport::includeSubroutine()   { wantSubroutine = true;  }
void ELfwkJobReport::suppressSubroutine()  { wantSubroutine = false; }

void ELfwkJobReport::includeText()   { wantText = true;  }
void ELfwkJobReport::suppressText()  { wantText = false; }

void ELfwkJobReport::includeContext()   { wantSomeContext = true;  }
void ELfwkJobReport::suppressContext()  { wantSomeContext = false; }

void ELfwkJobReport::suppressSerial()  { wantSerial = false; }
void ELfwkJobReport::includeSerial()   { wantSerial = true;  }

void ELfwkJobReport::useFullContext()  { wantFullContext = true;  }
void ELfwkJobReport::useContext()      { wantFullContext = false; }

void ELfwkJobReport::separateTime()  { wantTimeSeparate = true;  }
void ELfwkJobReport::attachTime()    { wantTimeSeparate = false; }

void ELfwkJobReport::separateEpilogue()  { wantEpilogueSeparate = true;  }
void ELfwkJobReport::attachEpilogue()    { wantEpilogueSeparate = false; }


// ----------------------------------------------------------------------
// Summary output:
// ----------------------------------------------------------------------

void ELfwkJobReport::summarization(
  const ELstring & fullTitle
, const ELstring & sumLines
)  {
  const int titleMaxLength( 40 );

  // title:
  //
  ELstring title( fullTitle, 0, titleMaxLength );
  int q = (lineLength - title.length() - 2) / 2;
  ELstring line(q, '=');
  emitToken( "", true );
  emitToken( line );
  emitToken( " " );
  emitToken( title );
  emitToken( " " );
  emitToken( line, true );

  // body:
  //
  *os << sumLines;

  // finish:
  //
  emitToken( "", true );
  emitToken( ELstring(lineLength, '='), true );

}  // summarization()


// ----------------------------------------------------------------------
// Changing ostream:
// ----------------------------------------------------------------------

void ELfwkJobReport::changeFile (std::ostream & os_) {
  os.reset(&os_, do_nothing_deleter());
  emitToken( "\n=======================================================", true );
  emitToken( "\nError Log changed to this stream\n" );
  emitToken( "\n=======================================================\n", true );
}

void ELfwkJobReport::changeFile (const ELstring & filename) {
  os.reset(new std::ofstream( filename.c_str(), std::ios/*_base*/::app ), close_and_delete());
  emitToken( "\n=======================================================", true );
  emitToken( "\nError Log changed to this file\n" );
  emitToken( "\n=======================================================\n", true );
}

void ELfwkJobReport::flush()  {
  os->flush();
}


// ----------------------------------------------------------------------


} // end of namespace service  
} // end of namespace edm  
