#ifndef ELCONTEXTSUPPLIER_H
#define ELCONTEXTSUPPLIER_H


// ----------------------------------------------------------------------
//
// ELcontextSupplier	a class with a few (pure virtual) methods to
//			provides strings for summary, ordinary, and
//			more verbose full constexts.  The context is
//			meant to convey framework-wide info, such as
//			current run and event.
//
//
//	THIS HEADER FILE DEFINES AN INTERFACE AND IS INCLUDED IN
//	FRAMEWORK CODE THAT OUGHT NOT TO BE FORCED TO RECOMPILE
//	UNNECESSARILY.
//
//	THEREFORE, CHANGES IN THIS FILE SHOULD BE AVOIDED IF POSSIBLE.
//
// 7/7/98 mf	Created file.
// 7/14/98 pgc  Renamed from ELcontextSupplier to ELcontextSupplier
// 9/8/98 web	Minor touch-ups
// 12/20/99 mf  Added virtual destructor.
//
// ----------------------------------------------------------------------


#ifndef ELSTRING_H
  #include "FWCore/MessageLogger/interface/ELstring.h"
#endif


namespace edm {       


// ----------------------------------------------------------------------
// prerequisite class:
// ----------------------------------------------------------------------

class ErrorObj;


// ----------------------------------------------------------------------
// ELcontextSupplier:
// ----------------------------------------------------------------------

class ELcontextSupplier  {

public:
  virtual ELcontextSupplier * clone()          const = 0;
  virtual ELstring            context()        const = 0;
  virtual ELstring            summaryContext() const = 0;
  virtual ELstring            fullContext()    const = 0;

  virtual void editErrorObj( ErrorObj & msg ) const  { }
  virtual ELstring traceRoutine( ) const  { return ELstring(""); }

  virtual ~ELcontextSupplier()  { ; }

};  // ELcontextSupplier


// ----------------------------------------------------------------------


}        // end of namespace edm


#endif  // ELCONTEXTSUPPLIER_H
