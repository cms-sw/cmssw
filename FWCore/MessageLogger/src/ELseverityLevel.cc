// ----------------------------------------------------------------------
//
// ELseverityLevel.cc - implement objects that encode a message's urgency
//
//      Both frameworker and user will often pass one of the
//      instantiated severity levels to logger methods.
//
//      The only other methods of ELseverityLevel a frameworker
//      might use is to check the relative level of two severities
//      using operator<() or the like.
//
// 29-Jun-1998 mf       Created file.
// 26-Aug-1998 WEB      Made ELseverityLevel object less weighty.
// 16-Jun-1999 mf       Added constructor from string, plus two lists
//                      of names to match.  Also added default constructor,
//                      more streamlined than default lev on original.
// 23-Jun-1999 mf       Modifications to properly handle pre-main order
//                      of initialization issues:
//                              Instantiation ofthe 14 const ELseverity &'s
//                              Instantiation of objectsInitialized as false
//                              Constructor of ELinitializeGlobalSeverityObjects
//                              Removed guarantor function in favor of the
//                              constructor.
// 30-Jun-1999 mf       Modifications to eliminate propblems with order of
//                      globals initializations:
//                              Constructor from lev calls translate()
//                              Constructor from string uses translate()
//                              translate() method
//                              List of strings for names in side getname() etc.
//                              Immediate initilization of ELsevLevGlobals
//                              Mods involving ELinitializeGlobalSeverityObjects
// 12-Jun-2000 web      Final fix to global static initialization problem
// 27-Jun-2000 web      Fix order-of-static-destruction problem
// 24-Aug-2000 web      Fix defective C++ switch generation
// 13-Jun-2007 mf       Change (requested by CMS) the name Severe to System
//			(since that his how MessageLogger uses that level)
// 21-Apr-2009 mf	Change the symbol for ELsev_success (which is used
//                      by CMS for LogDebug) from -! to -d.  
// ----------------------------------------------------------------------

#include <ostream>

#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ELmap.h"


// Possible Traces
// #define ELsevConTRACE


namespace edm
{


// ----------------------------------------------------------------------
// Helper to construct the string->ELsev_ map on demand:
// ----------------------------------------------------------------------
  
  typedef std::map< ELstring const, ELseverityLevel::ELsev_ > ELmap;
  
  static
  ELmap const &  loadMap()  {
    
    static ELmap  m;
    
    
    m[ ELzeroSeverity.getSymbol()   ] = ELseverityLevel::ELsev_zeroSeverity
    , m[ ELzeroSeverity.getName()     ] = ELseverityLevel::ELsev_zeroSeverity
    , m[ ELzeroSeverity.getInputStr() ] = ELseverityLevel::ELsev_zeroSeverity
    , m[ ELzeroSeverity.getVarName()  ] = ELseverityLevel::ELsev_zeroSeverity
    ;
    
    m[ ELdebug.getSymbol()   ] = ELseverityLevel::ELsev_success
    , m[ ELdebug.getName()     ] = ELseverityLevel::ELsev_success
    , m[ ELdebug.getInputStr() ] = ELseverityLevel::ELsev_success
    , m[ ELdebug.getVarName()  ] = ELseverityLevel::ELsev_success
    ;
    
    m[ ELinfo.getSymbol()   ] = ELseverityLevel::ELsev_info
    , m[ ELinfo.getName()     ] = ELseverityLevel::ELsev_info
    , m[ ELinfo.getInputStr() ] = ELseverityLevel::ELsev_info
    , m[ ELinfo.getVarName()  ] = ELseverityLevel::ELsev_info
    ;
    
    m[ ELwarning.getSymbol()   ] = ELseverityLevel::ELsev_warning
    , m[ ELwarning.getName()     ] = ELseverityLevel::ELsev_warning
    , m[ ELwarning.getInputStr() ] = ELseverityLevel::ELsev_warning
    , m[ ELwarning.getVarName()  ] = ELseverityLevel::ELsev_warning
    ;
    
    m[ ELerror.getSymbol()   ] = ELseverityLevel::ELsev_error
    , m[ ELerror.getName()     ] = ELseverityLevel::ELsev_error
    , m[ ELerror.getInputStr() ] = ELseverityLevel::ELsev_error
    , m[ ELerror.getVarName()  ] = ELseverityLevel::ELsev_error
    ;
    
    m[ ELunspecified.getSymbol()   ] = ELseverityLevel::ELsev_unspecified
    , m[ ELunspecified.getName()     ] = ELseverityLevel::ELsev_unspecified
    , m[ ELunspecified.getInputStr() ] = ELseverityLevel::ELsev_unspecified
    , m[ ELunspecified.getVarName()  ] = ELseverityLevel::ELsev_unspecified
    ;
    
    m[ ELsevere.getSymbol()   ] = ELseverityLevel::ELsev_severe
    , m[ ELsevere.getName()     ] = ELseverityLevel::ELsev_severe
    , m[ ELsevere.getInputStr() ] = ELseverityLevel::ELsev_severe
    , m[ ELsevere.getVarName()  ] = ELseverityLevel::ELsev_severe
    ;
    
    m[ ELhighestSeverity.getSymbol()   ] = ELseverityLevel::ELsev_highestSeverity
    , m[ ELhighestSeverity.getName()     ] = ELseverityLevel::ELsev_highestSeverity
    , m[ ELhighestSeverity.getInputStr() ] = ELseverityLevel::ELsev_highestSeverity
    , m[ ELhighestSeverity.getVarName()  ] = ELseverityLevel::ELsev_highestSeverity
    ;
    
    return m;
    
  }

// ----------------------------------------------------------------------
// Birth/death:
// ----------------------------------------------------------------------

ELseverityLevel::ELseverityLevel( enum ELsev_ lev ) : myLevel( lev )  {

  #ifdef ELsevConTRACE
    std::cerr << "--- ELseverityLevel " << lev
              << " (" << getName() << ")\n"
              << std::flush;
  #endif

}


ELseverityLevel::ELseverityLevel( ELstring const & s )  {
  
  static ELmap const & m = loadMap();
  
  ELmap::const_iterator  i = m.find( s );
  myLevel = ( i == m.end() ) ? ELsev_unspecified : i->second;
  
}

ELseverityLevel::~ELseverityLevel()  { ; }


// ----------------------------------------------------------------------
// Comparator:
// ----------------------------------------------------------------------

int  ELseverityLevel::cmp( ELseverityLevel const & e ) const  {
  return myLevel - e.myLevel;
}


// ----------------------------------------------------------------------
// Accessors:
// ----------------------------------------------------------------------

int  ELseverityLevel::getLevel()  const  {
  return  myLevel;
}


const ELstring  ELseverityLevel::getSymbol() const  {
  ELstring  result;

  switch ( myLevel )  {
    default                   :  result =  "0" ; break;
    case ELsev_zeroSeverity   :  result =  "--"; break;
    case ELsev_success        :  result =  "-d"; break;		// 4/21/09 mf
    case ELsev_info           :  result =  "-i"; break;
    case ELsev_warning        :  result =  "-w"; break;
    case ELsev_error          :  result =  "-e"; break;
    case ELsev_unspecified    :  result =  "??"; break;
    case ELsev_severe         :  result =  "-s"; break;
    case ELsev_highestSeverity:  result =  "!!"; break;
  };  // switch

  return result;
}


const ELstring  ELseverityLevel::getName() const  {
  ELstring  result;

  switch ( myLevel )  {
    default                   :  result =  "?no value?"; break;
    case ELsev_zeroSeverity   :  result =  "--"        ; break;
    case ELsev_success        :  result =  "Debug"     ; break; // 4/21/09 mf
    case ELsev_info           :  result =  "Info"      ; break;
    case ELsev_warning        :  result =  "Warning"   ; break;
    case ELsev_error          :  result =  "Error"     ; break;
    case ELsev_unspecified    :  result =  "??"        ; break;
    case ELsev_severe         :  result =  "System"    ; break; // 6/13/07 mf
    case ELsev_highestSeverity:  result =  "!!"        ; break;
  };  // switch

  return result;
}


const ELstring  ELseverityLevel::getInputStr() const  {
  ELstring  result;

  switch ( myLevel )  {
    default                   : result =  "?no value?" ; break;
    case ELsev_zeroSeverity   : result =  "ZERO"       ; break;
    case ELsev_success        : result =  "DEBUG"      ; break;
    case ELsev_info           : result =  "INFO"       ; break;
    case ELsev_warning        : result =  "WARNING"    ; break;
    case ELsev_error          : result =  "ERROR"      ; break;
    case ELsev_unspecified    : result =  "UNSPECIFIED"; break;
    case ELsev_severe         : result =  "SYSTEM"     ; break;  // 6/13/07 mf
    case ELsev_highestSeverity: result =  "HIGHEST"    ; break;
  };  // switch

  return result;
}


const ELstring  ELseverityLevel::getVarName() const  {
  ELstring  result;

  switch ( myLevel )  {
    default                   : result =  "?no value?       "; break;
    case ELsev_zeroSeverity   : result =  "ELzeroSeverity   "; break;
    case ELsev_success        : result =  "ELdebug          "; break;// 4/21/09
    case ELsev_info           : result =  "ELinfo           "; break;
    case ELsev_warning        : result =  "ELwarning        "; break;
    case ELsev_error          : result =  "ELerror          "; break;
    case ELsev_unspecified    : result =  "ELunspecified    "; break;
    case ELsev_severe         : result =  "ELsystem         "; break;// 6/13/07
    case ELsev_highestSeverity: result =  "ELhighestSeverity"; break;
  };  // switch

  return result;
}


// ----------------------------------------------------------------------
// Emitter:
// ----------------------------------------------------------------------

std::ostream & operator<<( std::ostream & os, const ELseverityLevel & sev )  {
  return os << " -" << sev.getName() << "- ";
}


// ----------------------------------------------------------------------
// Declare the globally available severity objects,
// one generator function and one proxy per non-default ELsev_:
// ----------------------------------------------------------------------

ELseverityLevel const  ELzeroSeverityGen()  {
  static ELseverityLevel const  e( ELseverityLevel::ELsev_zeroSeverity );
  return e;
}
ELslProxy< ELzeroSeverityGen    > const  ELzeroSeverity;

ELseverityLevel const  ELdebugGen()  {
  static ELseverityLevel const  e( ELseverityLevel::ELsev_success );
  return e;
}
ELslProxy< ELdebugGen         > const  ELdebug;

ELseverityLevel const  ELinfoGen()  {
  static ELseverityLevel const  e( ELseverityLevel::ELsev_info );
  return e;
}
ELslProxy< ELinfoGen            > const  ELinfo;

ELseverityLevel const  ELwarningGen()  {
  static ELseverityLevel const  e( ELseverityLevel::ELsev_warning );
  return e;
}
ELslProxy< ELwarningGen         > const  ELwarning;

ELseverityLevel const  ELerrorGen()  {
  static ELseverityLevel const  e( ELseverityLevel::ELsev_error );
  return e;
}
ELslProxy< ELerrorGen           > const  ELerror;

ELseverityLevel const  ELunspecifiedGen()  {
  static ELseverityLevel const  e( ELseverityLevel::ELsev_unspecified );
  return e;
}
ELslProxy< ELunspecifiedGen     > const  ELunspecified;

ELseverityLevel const  ELsevereGen()  {
  static ELseverityLevel const  e( ELseverityLevel::ELsev_severe );
  return e;
}
ELslProxy< ELsevereGen          > const  ELsevere;

ELseverityLevel const  ELhighestSeverityGen()  {
  static ELseverityLevel const  e( ELseverityLevel::ELsev_highestSeverity );
  return e;
}
ELslProxy< ELhighestSeverityGen > const  ELhighestSeverity;

// ----------------------------------------------------------------------


} // end of namespace edm  */
