#ifndef MessageLogger_ELseverityLevel_h
#define MessageLogger_ELseverityLevel_h


// ----------------------------------------------------------------------
//
// ELseverityLevel.h - declare objects that encode a message's urgency
//
//	Both frameworker and user will often pass one of the
//	instantiated severity levels to logger methods.
//
//	The only other methods of ELseverityLevel a frameworker
//	might use is to check the relative level of two severities
//	using operator< or the like.
//
// 30-Jun-1998 mf	Created file.
// 26-Aug-1998 WEB	Made ELseverityLevel object less weighty.
// 16-Jun-1999 mf	Added constructor from string.
// 23-Jun-1999 mf	Additional ELsev_noValueAssigned to allow constructor
//			from string to give ELunspecified when not found, while
//			still allowing finding zero severity.
// 23-Jun-1999 mf	Corrections for subtleties in initialization of
//			global symbols:
//				Added ELsevLevGlobals array
//				Changed extern consts of SLseverityLevels into
//				  const ELseverityLevel & 's
//				Inserted class ELinitializeGlobalSeverityObjects
//				  in place of the
//				  initializeGlobalSeverityObjects() function.
//				Changed globalSeverityObjectsGuarantor to an
//				  ELinitializeGlobalSeverityObjects instance.
// 30-Jun-1999 mf	Modifications to eliminate problems with order of
//                      globals initializations:
//				translate(), getInputStr(), getVarName()
// 12-Jun-2000 web	Final fix to global static initialization problem
// 14-Jun-2000 web	Declare classes before granting friendship.
// 27-Jun-2000 web	Fix order-of-static-destruction problem
//
// ----------------------------------------------------------------------


#ifndef ELSTRING_H
  #include "FWCore/MessageLogger/interface/ELstring.h"
#endif

namespace edm {       


// ----------------------------------------------------------------------
// Forward declaration:
// ----------------------------------------------------------------------

class ELseverityLevel;


// ----------------------------------------------------------------------
// Synonym for type of ELseverityLevel-generating function:
// ----------------------------------------------------------------------

typedef  ELseverityLevel const  ELslGen();


// ----------------------------------------------------------------------
// ELslProxy class template:
// ----------------------------------------------------------------------

template< ELslGen ELgen >
struct ELslProxy  {

  // --- birth/death:
  //
  ELslProxy();
  ~ELslProxy();

  // --- copying:
  //
  ELslProxy( ELslProxy const & );
  ELslProxy const &  operator= ( ELslProxy const & );

  // --- conversion::
  //
  operator ELseverityLevel const () const;

  // --- forwarding:
  //
  int             getLevel()    const;
  const ELstring  getSymbol()   const;
  const ELstring  getName()     const;
  const ELstring  getInputStr() const;
  const ELstring  getVarName()  const;

};  // ELslProxy<ELslGen>

// ----------------------------------------------------------------------


// ----------------------------------------------------------------------
// ELseverityLevel:
// ----------------------------------------------------------------------

class ELseverityLevel  {

public:

  // ---  One ELseverityLevel is globally instantiated (see below)
  // ---  for each of the following levels:
  //
  enum ELsev_  {
    ELsev_noValueAssigned = 0  // default returned by map when not found
  , ELsev_zeroSeverity         // threshold use only
  , ELsev_success              // report reaching a milestone
  , ELsev_info                 // information
  , ELsev_warning              // warning
  , ELsev_error                // error detected
  , ELsev_unspecified          // severity was not specified
  , ELsev_severe               // future results are suspect
  , ELsev_highestSeverity      // threshold use only
  // -----
  , nLevels                    // how many levels?
  };  // ELsev_

  // -----  Birth/death:
  //
  ELseverityLevel( ELsev_ lev = ELsev_unspecified );
  ELseverityLevel ( ELstring const & str );
        // str may match getSymbol, getName, getInputStr,
        // or getVarName -- see accessors
  ~ELseverityLevel();

  // -----  Comparator:
  //
  int  cmp( ELseverityLevel const & e ) const;

  // -----  Accessors:
  //
  int             getLevel()    const;
  const ELstring  getSymbol()   const;  // example: "-e"
  const ELstring  getName()     const;  // example: "Error"
  const ELstring  getInputStr() const;  // example: "ERROR"
  const ELstring  getVarName()  const;  // example: "ELerror"

  // -----  Emitter:
  //
  friend std::ostream &  operator<< (
    std::ostream          &  os
  , const ELseverityLevel &  sev
  );

private:

  // Data per ELseverityLevel object:
  //
  int    myLevel;

};  // ELseverityLevel


// ----------------------------------------------------------------------
// Declare the globally available severity objects,
// one generator function and one proxy per non-default ELsev_:
// ----------------------------------------------------------------------

extern ELslGen  ELzeroSeverityGen;
extern ELslProxy< ELzeroSeverityGen    > const  ELzeroSeverity;

extern ELslGen  ELdebugGen;
extern ELslProxy< ELdebugGen         > const  ELdebug;

extern ELslGen  ELinfoGen;
extern ELslProxy< ELinfoGen            > const  ELinfo;

extern ELslGen  ELwarningGen;
extern ELslProxy< ELwarningGen         > const  ELwarning;

extern ELslGen  ELerrorGen;
extern ELslProxy< ELerrorGen           > const  ELerror;

extern ELslGen  ELunspecifiedGen;
extern ELslProxy< ELunspecifiedGen     > const  ELunspecified;

extern ELslGen  ELsevereGen;
extern ELslProxy< ELsevereGen          > const  ELsevere;

extern ELslGen  ELhighestSeverityGen;
extern ELslProxy< ELhighestSeverityGen > const  ELhighestSeverity;


// ----------------------------------------------------------------------
// Comparators:
// ----------------------------------------------------------------------

extern bool  operator== ( ELseverityLevel const & e1
                        , ELseverityLevel const & e2 );
extern bool  operator!= ( ELseverityLevel const & e1
                        , ELseverityLevel const & e2 );
extern bool  operator<  ( ELseverityLevel const & e1
                        , ELseverityLevel const & e2 );
extern bool  operator<= ( ELseverityLevel const & e1
                        , ELseverityLevel const & e2 );
extern bool  operator>  ( ELseverityLevel const & e1
                        , ELseverityLevel const & e2 );
extern bool  operator>= ( ELseverityLevel const & e1
                        , ELseverityLevel const & e2 );


// ----------------------------------------------------------------------

}        // end of namespace edm


// ----------------------------------------------------------------------

#define ELSEVERITYLEVEL_ICC
  #include "FWCore/MessageLogger/interface/ELseverityLevel.icc"
#undef  ELSEVERITYLEVEL_ICC


// ----------------------------------------------------------------------

#endif  // MessageLogger_ELseverityLevel_h

