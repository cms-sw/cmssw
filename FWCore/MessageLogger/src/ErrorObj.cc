// ----------------------------------------------------------------------
//
// ErrorObj.cc
//
// History:
//
// Created 7/8/98 mf
// 6/16/99  mf, jvr     ErrorObj::operator<<( void (* f)(ErrorLog &) )
//                      allows an "ErrorObj << endmsg;"
// 7/16/99  jvr         Added setSeverity() and setID functions
// 6/7/00   web         Forbid setting extreme severities; guard against
//                      too-long ID's
// 9/21/00  mf          Added copy constructor for use by ELerrorList
// 5/7/01   mf          operator<< (const char[])
// 6/5/01   mf          setReactedTo
// 11/01/01 web         maxIDlength now unsigned
//
// 2/14/06  mf		Removed oerator<<(endmsg) which is not needed for
//			MessageLogger for CMS
//
// ErrorObj( const ELseverityLevel & sev, const ELstring & id )
// ~ErrorObj()
// set( const ELseverityLevel & sev, const ELstring & id )
// clear()
// setProcess   ( const ELstring & proc )
// setModule    ( const ELstring & module )
// setSubroutine( const ELstring & subroutine )
// emit( const ELstring & txt )
// operator<<( void (* f)(ErrorLog &) )
//
// ----------------------------------------------------------------------


#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"
#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELlist.h"

#ifndef IOSTREAM_INCLUDED
#endif


// ----------------------------------------------------------------------


// Possible Traces
// #define ErrorObjCONSTRUCTOR_TRACE
// #define ErrorObj_EMIT_TRACE
// #define ErrorObj_SUB_TRACE


// ----------------------------------------------------------------------


namespace edm
{


// ----------------------------------------------------------------------
// Class static and class-wide parameter:
// ----------------------------------------------------------------------

int  ErrorObj::ourSerial(  0 );
const unsigned int  maxIDlength( 20 );


// ----------------------------------------------------------------------
// Birth/death:
// ----------------------------------------------------------------------

ErrorObj::ErrorObj( const ELseverityLevel & sev, const ELstring & id )  {

  #ifdef ErrorObjCONSTRUCTOR_TRACE
    std::cerr << "Constructor for ErrorObj\n";
  #endif

  clear();
  set( sev, id );

}  // ErrorObj()


ErrorObj::ErrorObj( const ErrorObj & orig )  :
        mySerial        ( orig.mySerial ),
        myXid           ( orig.myXid ),
        myIdOverflow    ( orig.myIdOverflow ),
        myTimestamp     ( orig.myTimestamp ),
        myItems         ( orig.myItems ),
        myReactedTo     ( orig.myReactedTo )
{

  #ifdef ErrorObjCONSTRUCTOR_TRACE
    std::cerr << "Copy Constructor for ErrorObj\n";
  #endif

}  // ErrorObj(ErrorObj)


ErrorObj::~ErrorObj()  {

  #ifdef ErrorObjCONSTRUCTOR_TRACE
    std::cerr << "Destructor for ErrorObj\n";
  #endif

}  // ~ErrorObj()


// ----------------------------------------------------------------------
// Accessors:
// ----------------------------------------------------------------------

int                   ErrorObj::serial()     const  { return mySerial; }
const ELextendedID &  ErrorObj::xid()        const  { return myXid; }
const ELstring &      ErrorObj::idOverflow() const  { return myIdOverflow; }
time_t                ErrorObj::timestamp()  const  { return myTimestamp; }
const ELlist_string & ErrorObj::items()      const  { return myItems; }
bool                  ErrorObj::reactedTo()  const  { return myReactedTo; }


ELstring ErrorObj::context() const {
  return myContext;
}

ELstring ErrorObj::fullText() const  {

  ELstring result;
  for ( ELlist_string::const_iterator it = myItems.begin();
        it != myItems.end();
        ++it )
    result +=  *it;
  return result;

}  // fullText()


// ----------------------------------------------------------------------
// Mutators:
// ----------------------------------------------------------------------

void ErrorObj::setSeverity( const ELseverityLevel & sev )  {
  myXid.severity = (sev <= ELzeroSeverity   ) ? (ELseverityLevel)ELincidental
                 : (sev >= ELhighestSeverity) ? (ELseverityLevel)ELfatal
                 :                              sev
                 ;
}


void ErrorObj::setID( const ELstring & id )  {
  myXid.id = ELstring( id, 0, maxIDlength );
  if ( id.length() > maxIDlength )
    myIdOverflow = ELstring( id, maxIDlength, id.length()-maxIDlength );
}


void ErrorObj::setModule( const ELstring & module )  { myXid.module = module; }

void ErrorObj::setContext( const ELstring & c )  { myContext = c; }


void ErrorObj::setSubroutine( const ELstring & subroutine )  {
  #ifdef ErrorObj_SUB_TRACE
    std::cerr << "=:=:=: ErrorObj::setSubroutine(" << subroutine << ")\n";
  #endif
  myXid.subroutine = (subroutine[0] == ' ')
                   ? subroutine.substr(1)
                   : subroutine;
}


void ErrorObj::setProcess( const ELstring & proc )  {
  myXid.process = proc;
  #if 0
    std::cerr << "ErrorObj process set to \"" << proc << "\"\n";
  #endif
}

void ErrorObj::setReactedTo( bool r )  {
  myReactedTo = r;
}


#ifdef ErrorObj_SUB_TRACE
  static int subN = 0;
#endif


ErrorObj & ErrorObj::emit( const ELstring & s )  {

  #ifdef ErrorObj_EMIT_TRACE
    std::cerr << "=:=:=: ErrorObj::emit ( " << s << " )\n";
  #endif

  #ifdef ErrorObj_SUB_TRACE
    if ( subN > 0 )  {
      std::cerr << "=:=:=: subN ErrorObj::emit ( " << s << " )\n";
      subN--;
    }
  #endif

  if ( eq_nocase(s.substr(0,5), "@SUB=" ) )  {
    #ifdef ErrorObj_SUB_TRACE
      std::cerr << "=:=:=: ErrorObj::@SUB s.substr(5) is: " << s.substr(5)
                << '\n';
    #endif
    setSubroutine(s.substr(5));
  }
  else  {
    myItems.push_back( s );
  }

  return * this;

}  // emit()


void ErrorObj::set( const ELseverityLevel & sev, const ELstring & id )  {

  clear();

  myTimestamp = time( 0 );
  mySerial = ++ ourSerial;

  setID( id );
  setSeverity( sev );

}  // set()


void ErrorObj::clear()  {

  mySerial     = 0;
  myXid.clear();
  myIdOverflow = "";
  myTimestamp  = 0;
  myItems.erase( myItems.begin(), myItems.end() );  // myItems.clear();
  myReactedTo  = false;

}  // clear()


ErrorObj & operator<<( ErrorObj & e, const char s[] ) {
  // Exactly equivalent to the general template.
  // If this is not provided explicitly, then the template will
  // be instantiated once for each length of string ever used.
  std::ostringstream  ost;
  ost << s << ' ';
  return  e.emit( ost.str() );
}


} // end of namespace edm  */
