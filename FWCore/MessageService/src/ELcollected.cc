// ----------------------------------------------------------------------
//
// ELcollected.cc
//
// History:
//   07-Jun-2000  WEB  Reflect consolidation of ELdestination/X and of
//     ELoutput/X; consolidate ELcollected/X
//   15-Sep-2000  MF   Corrected bug:  copy constructor needs to set sender
//                      based on a clone() of *sender, not a bit copy, since
//                      the whole reasoning was to protect against the user's
//                      objects going out of scope.
//   4-Apr-2001   MF    Loginc to ignore ore respond to modules, using base
//                      method thisShouldBeIgnored().
//
// ----------------------------------------------------------------------

#include "FWCore/MessageService/interface/ELcollected.h"
#include "FWCore/MessageService/interface/ELdestination.h"
#include "FWCore/MessageService/interface/ELadministrator.h"
#include "FWCore/MessageService/interface/ELcontextSupplier.h"

#include <sstream>

// Possible Traces:
//#define ELcollectedCONSTRUCTOR_TRACE
//#define ELcollectedTRACE_LOG
//#define ELcollected_EMIT_TRACE

namespace edm {
namespace service {


// ----------------------------------------------------------------------
// Constructors:
// ----------------------------------------------------------------------

ELcollected::ELcollected( const ELsender & snd )
: ELoutput()
, sender  ( snd.clone() )
{

  #ifdef ELcollectedCONSTRUCTOR_TRACE
    std::cout << "Constructor for ELcollected(ELsender)\n";
  #endif

  // Unlike ELoutput, we do not emit Error Log established message.

}  // ELcollected()


ELcollected::ELcollected( const ELcollected & orig )
: ELoutput( orig        )
, sender  ( orig.sender->clone() )              // $$ mf 9/15/00
{

  #ifdef ELcollectedCONSTRUCTOR_TRACE
    std::cout << "Copy constructor for ELcollected\n";
  #endif

  ignoreMostModules    = orig.ignoreMostModules;
  respondToThese       = orig.respondToThese;
  respondToMostModules = orig.respondToMostModules;
  ignoreThese          = orig.ignoreThese;

}  // ELcollected()


ELcollected::~ELcollected()  {

  #ifdef ELcollectedCONSTRUCTOR_TRACE
    std::cout << "Destructor for ELcollected)\n";
  #endif
}  // ~ELcollected()



// ----------------------------------------------------------------------
// Methods invoked by the ELadministrator:
// ----------------------------------------------------------------------

ELcollected *
ELcollected::clone() const  {

  return  new ELcollected( *this );

} // clone()


// Remainder are from base class.

// =======
// intoBuf
// =======

void ELcollected::intoBuf( const ELstring & s )  {

  buf += s;
  buf += '\0';

} // intoBuf();


// ========
// emitXid
// ========

void ELcollected::emitXid( const ELextendedID & xid )  {

  buf = "";

  intoBuf ( xid.process    );
  intoBuf ( xid.module     );
  intoBuf ( xid.subroutine );
  intoBuf ( xid.id         );
  std::ostringstream ost;
  ost << xid.severity.getLevel();
  intoBuf ( ost.str() );

} // emitXid


// =====
// emit
// =====

void ELcollected::emit( const ELstring & s, bool nl )  {

  #ifdef ELcollected_EMIT_TRACE
    std::cout << "[][][] in emit:  s.length() " << s.length() << "\n";
  #endif

  // A forced newline is something that needs to be transmitted.
  // We shall insert it as its own item.

  if (s.length() == 0)  {
    if ( nl )  {
      intoBuf( newline );
      charsOnLine = 0;
    }
    return;
  }

  // Setting up for indentation if you start with a nweline, or if the length
  // exceeds column 80, is not done here, it is done on the server.

  #ifdef ELcollected_EMIT_TRACE
    std::cout << "[][][] in emit: about to << s to buf: " << s << " \n";
  #endif

  // Place the item into the buffer

  intoBuf (s);

  // Accounting for trailing \n is done at the server.

  // A forced trailing newline is something that needs to be transmitted.
  // We shall insert it as its own item.

  if ( nl )  {
    intoBuf (newline);
  }

  #ifdef ELcollected_EMIT_TRACE
    std::cout << "[][][] in emit: completed \n";
  #endif

}  // emit()



// ====
// log
// ====

bool ELcollected::log( const edm::ErrorObj & msg )  {

  #ifdef ELcollectedTRACE_LOG
    std::cout << "    =:=:=: Log to an ELcollected \n";
  #endif

  xid = msg.xid();      // Save the xid.

  // See if this message is to be acted upon, and add it to limits table
  // if it was not already present:
  //
  if ( msg.xid().severity < threshold  )  return false;
  if ( thisShouldBeIgnored(xid.module) )  return false;
  if ( ! limits.add( msg.xid() )      )   return false;

  #ifdef ELcollectedTRACE_LOG
    std::cout << "    =:=:=: Limits table work done \n";
  #endif

  // start the buffer with the xid

  emitXid (xid);

  //

  #ifdef ELcollectedTRACE_LOG
    std::cout << "    =:=:=: xid emitted \n";
  #endif

  // Provide the context information.  The server side will use this to prime
  // its special context supplier.  We will send over all three types of
  // context, even though probably only 1 or 2 will be needed.

  emit( ELadministrator::instance()->
                        getContextSupplier().summaryContext());
  emit( ELadministrator::instance()->
                        getContextSupplier().context());
  emit( ELadministrator::instance()->
                        getContextSupplier().fullContext());

  #ifdef ELcollectedTRACE_LOG
    std::cout << "    =:=:=: Context done: \n";
  #endif

  // No prologue separate from what the server will issue.

  // No serial number of message separate from what the server will issue.

  // collected each item in the message:
  //
  if ( wantText )  {
    ELlist_string::const_iterator it;
    for ( it = msg.items().begin();  it != msg.items().end();  ++it )  {
  #ifdef ELcollectedTRACE_LOG
    std::cout << "      =:=:=: Item:  " << *it <<"\n";
  #endif
      emit( *it );
    }
  }

  // DO NOT Provide further identification such as module and subroutine;
  // the server side will provide that using the xid you have sent it, if
  // the server side user wants it.

  // DO NOT provide time stamp; it would duplicate server's stamp!

  // Provide traceback information:
  //
  if ( msg.xid().severity >= traceThreshold )  {
    emit( ELstring("\n")
          + ELadministrator::instance()->getContextSupplier().traceRoutine()
        , true );
  }
  else  {                                        //else statement added JV:1
    emit( "", true );
  }
  #ifdef ELcollectedTRACE_LOG
    std::cout << "    =:=:=: Trace routine done: \n";
  #endif

  // Message has been fully processed through emit; now put in an extra
  // zero, and send out the buffer.
  //

  buf += char(0);
  int nbuf = buf.length();

  sender->send ( nbuf, buf.data() );

  #ifdef ELcollectedTRACE_LOG
    std::cout << "  =:=:=: log(msg) done: \n";
  #endif

  return true;

}  // log()


// ----------------------------------------------------------------------


} // end of namespace service  
} // end of namespace edm  

