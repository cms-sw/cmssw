// ----------------------------------------------------------------------
//
// ELcontextSupplier.cc
//
//
// 12/20/05   jm, mf    Created, based on ELoutput.
//
// ----------------------------------------------------------------------


#include "EventFilter/Message2log4cplus/interface/ELlog4cplus.h"

#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageService/interface/ELadministrator.h"
#include "FWCore/MessageService/interface/ELcontextSupplier.h"

#include "log4cplus/logger.h"
#include "log4cplus/fileappender.h"
#include "log4cplus/loglevel.h"

#include "xdaq/Application.h"
#include <memory>

// Possible Traces:
// #define ELoutputCONSTRUCTOR_TRACE
//#define ELlog4cplusTRACE_LOG
//#define ELlog4cplus_EMIT_TRACE

#include <iostream>
#include <fstream>

namespace edm
{

namespace {
  void makeFileAppender()
  {
    static bool iscalled = false;
    if(iscalled) return;
    iscalled=true;

    using namespace log4cplus;
    using namespace log4cplus::helpers;

    SharedAppenderPtr ap(new FileAppender("log4cplus.output"));
    ap->setName("Main");
    ap->setLayout(std::auto_ptr<Layout>(new log4cplus::TTCCLayout()) );
    Logger::getRoot().addAppender(ap);
  }
}

// ----------------------------------------------------------------------
// Useful function:
// ----------------------------------------------------------------------


static char * formatTime( const time_t t )  {

static char ts[] = "dd-Mon-yyyy hh:mm:ss XYZ";


#ifdef ANALTERNATIVE
  char * c  = ctime( &t );                      // 6/14/99 mf Can't be static!
  strncpy( ts+ 0, c+ 8, 2 );  // dd
  strncpy( ts+ 3, c+ 4, 3 );  // Mon
  strncpy( ts+ 7, c+20, 4 );  // yyyy
  strncpy( ts+12, c+11, 8 );  // hh:mm:ss
  strncpy( ts+21, tzname[localtime(&t)->tm_isdst], 3 );  // CST
#endif

  strftime( ts, strlen(ts)+1, "%d-%b-%Y %H:%M:%S %Z", localtime(&t) );
                // mf 4-9-04


  return ts;

}  // formatTime()


// ----------------------------------------------------------------------
// Constructors:
// ----------------------------------------------------------------------

ELlog4cplus::ELlog4cplus()
: ELdestination       (            )
, os                  ( &os_ )
, osIsOwned           ( false      )
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
, xxxxInt             ( 0          )
, appl_               ( 0          )
{
  //  makeFileAppender(); // this is not needed/wanted. An appender must be provided by the application itself

  #ifdef ELlog4cplusCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ELlog4cplus()\n";
  #endif

  lineLength = 32000;

  emit( "\n=======================================================", true );
  emit( "\nMessageLogger service established\n" );
  emit( formatTime(time(0)), true );
  emit( "\n=======================================================\n", true );

}  // ELlog4cplus()



ELlog4cplus::ELlog4cplus( const ELlog4cplus & orig )
: ELdestination       (                           )
, os                  ( &os_                      )
, osIsOwned           ( orig.osIsOwned            )
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
, xxxxInt             ( orig.xxxxInt              )
, appl_               ( orig.appl_                ) 
{

  #ifdef ELlog4cplusCONSTRUCTOR_TRACE
    std::cerr << "Copy constructor for ELlog4cplus\n";
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

  // ownership, if any, passes to new copy:
  const_cast<ELlog4cplus &>(orig).osIsOwned = false;

}  // ELlog4cplus()


ELlog4cplus::~ELlog4cplus()  {

  #ifdef ELlog4cplusCONSTRUCTOR_TRACE
    std::cerr << "Destructor for ELlog4cplus\n";
  #endif

  if ( osIsOwned )  { // we have an ofstream
    ((std::ofstream*)os)->close();
    delete os;
  }

}  // ~ELlog4cplus()


// ----------------------------------------------------------------------
// Methods invoked by the ELadministrator:
// ----------------------------------------------------------------------

ELlog4cplus *
ELlog4cplus::clone() const  {

  return new ELlog4cplus( *this );

} // clone()


bool ELlog4cplus::log( const ErrorObj & msg )  {
  os->str(std::string());

  #ifdef ELlog4cplusTRACE_LOG
    std::cerr << "    =:=:=: Log to an ELlog4cplus \n";
  #endif

  xid = msg.xid();      // Save the xid.

  // See if this message is to be acted upon, and add it to limits table
  // if it was not already present:
  //
  if ( msg.xid().severity < threshold  )  return false;
  if ( thisShouldBeIgnored(xid.module) )  return false;
  if ( ! limits.add( msg.xid() )       )  return false;
  
#ifdef ELlog4cplusTRACE_LOG
  std::cerr << "    =:=:=: Limits table work done \n";
#endif
  
  // get log4cplus logger and establish (log4cplus) context 
  bool mustPop = false;

  log4cplus::Logger loghere = appl_ ? appl_->getApplicationLogger() :
    log4cplus::Logger::getInstance(msg.xid().module.c_str());
  LOG4CPLUS_DEBUG(loghere,  "Message2log4cplus will use logger from appl_ ? " 
		 << (appl_ ? "yes" : "no"));
  if(appl_)
    {
      log4cplus::getNDC().push(msg.xid().module.c_str());
      mustPop = true;
    }
  log4cplus::getNDC().push(msg.context().c_str());
  
  // Output the prologue:
  //
  emit( preamble );
  emit( xid.severity.getSymbol() );
  emit( " " );
  emit( xid.id );
  emit( msg.idOverflow() );
  emit( ": " );

  #ifdef ELlog4cplusTRACE_LOG
    std::cerr << "    =:=:=: Prologue done \n";
  #endif
  // Output serial number of message:
  //
  if ( wantSerial )  {
    std::ostringstream s;
    s << msg.serial();
    emit( "[serial #" + s.str() + ELstring("] ") );
  }

  // Output each item in the message:
  //
  if ( wantText )  {
    ELlist_string::const_iterator it;
    for ( it = msg.items().begin();  it != msg.items().end();  ++it )  {
    #ifdef ELlog4cplusTRACE_LOG
      std::cerr << "      =:=:=: Item:  " << *it << '\n';
    #endif
      emit( *it );
    }
  }

  // Provide further identification:
  //
  bool needAspace = true;
  if ( wantEpilogueSeparate )  {
    if ( xid.module.length() + xid.subroutine.length() > 0 )  {
      emit("\n");
      needAspace = false;
    }
    else if ( wantTimestamp && !wantTimeSeparate )  {
      emit("\n");
      needAspace = false;
    }
  }
  if ( wantModule && (xid.module.length() > 0) )  {
    if (needAspace) { emit(ELstring(" ")); needAspace = false; }
    emit( xid.module + ELstring(" ") );
  }
  if ( wantSubroutine && (xid.subroutine.length() > 0) )  {
    if (needAspace) { emit(ELstring(" ")); needAspace = false; }
    emit( xid.subroutine + "()" + ELstring(" ") );
  }

  #ifdef ELlog4cplusTRACE_LOG
    std::cerr << "    =:=:=: Module and Subroutine done \n";
  #endif

  // Provide time stamp:
  //
  if ( wantTimestamp )  {
    if ( wantTimeSeparate )  {
      emit( ELstring("\n") );
      needAspace = false;
    }
    if (needAspace) { emit(ELstring(" ")); needAspace = false; }
    emit( formatTime(msg.timestamp()) + ELstring(" ") );
  }

  #ifdef ELlog4cplusTRACE_LOG
    std::cerr << "    =:=:=: TimeStamp done \n";
  #endif

  // Provide the context information:
  //
  if ( wantSomeContext )
    if (needAspace) { emit(ELstring(" ")); needAspace = false; }
    #ifdef ELlog4cplusTRACE_LOG
      std::cerr << "    =:=:=:>> context supplier is at 0x"
                << std::hex
                << &service::ELadministrator::instance()->getContextSupplier() << '\n';
      std::cerr << "    =:=:=:>> context is --- "
                << service::ELadministrator::instance()->getContextSupplier().context()
                << '\n';
    #endif
    if ( wantFullContext )  {
      emit( service::ELadministrator::instance()->getContextSupplier().fullContext());
    #ifdef ELlog4cplusTRACE_LOG
      std::cerr << "    =:=:=: fullContext done: \n";
    #endif
    } else  {
      emit( service::ELadministrator::instance()->getContextSupplier().context());
  #ifdef ELlog4cplusTRACE_LOG
    std::cerr << "    =:=:=: Context done: \n";
  #endif
    }

  // Provide traceback information:
  //
  if ( msg.xid().severity >= traceThreshold )  {
    emit( ELstring("\n")
          + service::ELadministrator::instance()->getContextSupplier().traceRoutine()
        , true );
  }
  else  {                                        //else statement added JV:1
    emit ("", true);
  }
  #ifdef ELlog4cplusTRACE_LOG
    std::cerr << "    =:=:=: Trace routine done: \n";
  #endif

  // Done; message has been fully processed:
  //

  #ifdef ELlog4cplusTRACE_LOG
    std::cerr << "  =:=:=: log(msg) done: \n";
  #endif

    // std::cout << os->str() << "\n";

    switch(msg.xid().severity.getLevel())
      {
      case edm::ELseverityLevel::ELsev_success:
	{
	  // success is used for debug here
	  LOG4CPLUS_DEBUG(loghere,os->str());
	  break;
	}
      case edm::ELseverityLevel::ELsev_info:
	{
	  LOG4CPLUS_INFO(loghere,os->str());
	  break;
	}
      case edm::ELseverityLevel::ELsev_warning:
	{
	  LOG4CPLUS_WARN(loghere,os->str());
	  break;
	}
      case edm::ELseverityLevel::ELsev_error:
      default:
	{
	  LOG4CPLUS_ERROR(loghere,os->str());
	  break;
	}
      }
    if(mustPop) log4cplus::getNDC().pop();
    log4cplus::getNDC().pop();
  return true;

}  // log()


// Remainder are from base class.


// ----------------------------------------------------------------------
// Maintenance and test functionality:
// ----------------------------------------------------------------------

void ELlog4cplus::xxxxSet( int i )  {
  xxxxInt = i;
}

void ELlog4cplus::xxxxShout()  {
  std::cerr << "XXXX ELlog4cplus: " << xxxxInt << std::endl;
}


// ----------------------------------------------------------------------
// Output methods:
// ----------------------------------------------------------------------

void ELlog4cplus::emit( const ELstring & s, bool nl )  {

  #ifdef ELlog4cplus_EMIT_TRACE
    std::cerr << "[][][] in emit:  charsOnLine is " << charsOnLine << '\n';
    std::cerr << "[][][] in emit:  s.length() " << s.length() << '\n';
    std::cerr << "[][][] in emit:  lineLength is " << lineLength << '\n';
  #endif

  if (s.length() == 0)  {
    if ( nl )  {
      (*os) << newline << std::flush;
      charsOnLine = 0;
    }
    return;
  }

  char first = s[0];
  char second,
       last,
       last2;
  second = (s.length() < 2) ? '\0' : s[1];
  last = (s.length() < 2) ? '\0' : s[s.length()-1];
  last2 = (s.length() < 3) ? '\0' : s[s.length()-2];
         //checking -2 because the very last char is sometimes a ' ' inserted
         //by ErrorLog::operator<<

               //Accounts for newline @ the beginning of the ELstring     JV:2
  if ( first == '\n'
  || (charsOnLine + static_cast<int>(s.length())) > lineLength )  {
    #ifdef ELlog4cplus_EMIT_TRACE
      std::cerr << "[][][] in emit: about to << to *os \n";
    #endif
    (*os) << newline << indent;
    charsOnLine = indent.length();
    if (second != ' ')  {
      (*os) << ' ';
      charsOnLine++;
    }
    if ( first == '\n' )  {
      (*os) << s.substr(1);
    }
    else  {
      (*os) << s;
    }
  }

  #ifdef ELlog4cplus_EMIT_TRACE
    std::cerr << "[][][] in emit: about to << s to *os: " << s << " \n";
  #endif

  else  {
    (*os) << s;
  }

  if (last == '\n' || last2 == '\n')  {  //accounts for newline @ end    $$ JV:2
    (*os) << indent;                    //of the ELstring
    if (last != ' ')
      (*os) << ' ';
    charsOnLine = indent.length() + 1;
  }

  if ( nl )  { (*os) << newline << std::flush; charsOnLine = 0;           }
  else       {                                 charsOnLine += s.length(); }

  #ifdef ELlog4cplus_EMIT_TRACE
    std::cerr << "[][][] in emit: completed \n";
  #endif

}  // emit()


// ----------------------------------------------------------------------
// Methods controlling message formatting:
// ----------------------------------------------------------------------

void ELlog4cplus::includeTime()   { wantTimestamp = true;  }
void ELlog4cplus::suppressTime()  { wantTimestamp = false; }

void ELlog4cplus::includeModule()   { wantModule = true;  }
void ELlog4cplus::suppressModule()  { wantModule = false; }

void ELlog4cplus::includeSubroutine()   { wantSubroutine = true;  }
void ELlog4cplus::suppressSubroutine()  { wantSubroutine = false; }

void ELlog4cplus::includeText()   { wantText = true;  }
void ELlog4cplus::suppressText()  { wantText = false; }

void ELlog4cplus::includeContext()   { wantSomeContext = true;  }
void ELlog4cplus::suppressContext()  { wantSomeContext = false; }

void ELlog4cplus::suppressSerial()  { wantSerial = false; }
void ELlog4cplus::includeSerial()   { wantSerial = true;  }

void ELlog4cplus::useFullContext()  { wantFullContext = true;  }
void ELlog4cplus::useContext()      { wantFullContext = false; }

void ELlog4cplus::separateTime()  { wantTimeSeparate = true;  }
void ELlog4cplus::attachTime()    { wantTimeSeparate = false; }

void ELlog4cplus::separateEpilogue()  { wantEpilogueSeparate = true;  }
void ELlog4cplus::attachEpilogue()    { wantEpilogueSeparate = false; }


// ----------------------------------------------------------------------
// Summary output:
// ----------------------------------------------------------------------

void ELlog4cplus::summarization(
  const ELstring & fullTitle
, const ELstring & sumLines
)  {
  const int titleMaxLength( 40 );

  // title:
  //
  ELstring title( fullTitle, 0, titleMaxLength );
  int q = (lineLength - title.length() - 2) / 2;
  ELstring line(q, '=');
  emit( "", true );
  emit( line );
  emit( " " );
  emit( title );
  emit( " " );
  emit( line, true );

  // body:
  //
  *os << sumLines;

  // finish:
  //
  emit( "", true );
  emit( ELstring(lineLength, '='), true );

}  // summarization()

void ELlog4cplus::setAppl(xdaq::Application *a)
{
  std::cout << "setting application pointer in ELlog4cplus" << std::endl;
  appl_ = a;
}

// ----------------------------------------------------------------------


} // end of namespace edm  */
