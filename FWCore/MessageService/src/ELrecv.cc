// ----------------------------------------------------------------------
//
// ELrecv.cc    This is the global method which the server-side program calls
//              when it receives a message which originated from ELsend on the
//              client process.
//
// This file defines ELrecv() and its pet context supplier ELservConSup class.
//
// ----------------------------------------------------------------------


#include "FWCore/MessageService/interface/ELrecv.h"
#include "FWCore/MessageService/interface/ErrorLog.h"
#include "FWCore/MessageService/interface/ELadministrator.h"
#include "FWCore/MessageService/interface/ELcontextSupplier.h"

#ifndef CSTRING_INCLUDED
  #include <cstring>
#endif


namespace edm {
namespace service {

class ELservConSup : public ELcontextSupplier  {

public:
  ELstring context()                      const override  { return con_; }
  ELstring summaryContext()               const override  { return sumcon_; }
  ELstring fullContext()                  const override  { return fullcon_; }
  void setContext        ( const ELstring & s )  { con_     = s; }
  void setSummaryContext ( const ELstring & s )  { sumcon_  = s; }
  void setFullContext    ( const ELstring & s )  { fullcon_ = s; }
  ELservConSup * clone() const override {  return new ELservConSup ( *this );   }
private:
  ELstring  con_;
  ELstring  sumcon_;
  ELstring  fullcon_;

};

void  ELrecv ( int nbytes, const char * data )  {
  ELrecv ( nbytes, data, "*ELrecv*" );
}


void  ELrecv ( int nbytes, const char * data, ELstring localModule )  {

  static const int  MAXITEMCOUNT = 100;

  static ErrorLog           errlog;
  static ELservConSup       con;
  static ELadministrator *  logger = ELadministrator::instance();

  // Pull out the extended id.  Carefully look at ELcollected::emitXid to
  // make certain we take things out in the proper order.

  ELstring process;
  ELstring module;
  ELstring subroutine;
  ELstring id;
  ELstring sevString;
  int      sevLevel;

  const char *  nextItem = data;

  process = nextItem;
  nextItem += strlen(nextItem) + 1;

  if (localModule == "*ELrecv*")  module = nextItem;
  else                            module = localModule + ":" + nextItem;
  nextItem += strlen(nextItem) + 1;

  subroutine = nextItem;
  nextItem += strlen(nextItem) + 1;

  id = nextItem;
  nextItem += strlen(nextItem) + 1;

  sevString = nextItem;
  nextItem += strlen(nextItem) + 1;
  std::istringstream  ist (sevString);
  if ( ! (ist >> sevLevel) )  {
    // This should not be possible.  But if it does, we don't want to
    // kill off the probably critical error monitoring job!
    errlog ( ELerror2, "Collection Error" ) <<
        "An error message received has an unreadable value of severity level"
        << sevString << endmsg;
    sevLevel = ELseverityLevel::ELsev_unspecified;
  }
  if ( sevLevel < ELseverityLevel::ELsev_zeroSeverity ||
       sevLevel > ELseverityLevel::ELsev_highestSeverity )  {
    // Again, this should not be possible.
    errlog ( ELerror2, "Collection Error" ) <<
        "An error message received has an out-of-range value of severity level"
        << sevString << endmsg;
    sevLevel = ELseverityLevel::ELsev_unspecified;
  }

  // Pull out the context strings and set up the special supplier.

  ELstring  context;

  context = nextItem;
  nextItem += strlen(nextItem) + 1;
  con.setSummaryContext( context );

  context = nextItem;
  nextItem += strlen(nextItem) + 1;
  con.setContext( context );

  context = nextItem;
  nextItem += strlen(nextItem) + 1;
  con.setFullContext( context );

  // Remember the context supplier, and substitute the special supplier.

  ELcontextSupplier & oldCS = logger->swapContextSupplier(con);

  // Set the module, subroutine, and process according to this xid.

  ELstring  oldProcess = logger->swapProcess(process);
  errlog.setModule (module);
  errlog.setSubroutine(subroutine);

  // Instantiate a message with the appropriate severity level and id.

  errlog ( ELseverityLevel( ELseverityLevel::ELsev_(sevLevel)), id );

  // Add the remaining items to this message.
  //    To avoid any possibility of a completely runaway message, we limit
  //    the number of items output to 100 by doing a for instead of a while
  //    loop.

  ELstring  item;
  int       itemCount;
  for ( itemCount = 0; itemCount < MAXITEMCOUNT; itemCount++ )  {
    if (*nextItem == 0) break;
    item = nextItem;
    nextItem += strlen(nextItem) + 1;
    errlog << item;
  }

  // If the direct ELrecv form was used, end the message,
  // thus causing the logging.  If the errlog (nbytes, data) form was
  // used, the user will supply the endmsg.

  if ( localModule == "*ELrecv*" )  {
    errlog << endmsg;
  }

  if ( *nextItem++ != 0 )  {
    errlog << endmsg;
    errlog ( ELerror2, "Collection Error" ) <<
        "Apparent runaway error message on the previous message!" <<
        "Truncated after " << MAXITEMCOUNT << "items." << endmsg;
  }
  int  diff = nextItem - data;
  if ( diff != nbytes )  {
    errlog << endmsg;
    errlog ( ELerror2, "Collection Error" ) <<
    "Apparent corrupted transmission of error message"
    << "on the previous message!"
    << "Total length was" << diff << "nbytes transmitted was" << nbytes
    << endmsg;
  }

  // End the message, before we swap back the context and process.
  // (This is what makes the endmsg in errlog(nbytes, data) << endmsg
  // superfluous.)

  errlog << endmsg;

  // Reset the context supplier and process to the remembered value.

  logger->swapContextSupplier(oldCS);
  logger->swapProcess(oldProcess);

} // ELrecv()


} // end of namespace service  
} // end of namespace edm  
