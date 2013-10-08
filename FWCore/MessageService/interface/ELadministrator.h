#ifndef MessageService_ELadministrator_h
#define MessageService_ELadministrator_h


// ----------------------------------------------------------------------
//
// ELadminstrator.h  provides the singleton class that the framework uses to
//		     control logger behavior including attaching destinations.
//		Includes the methods used by ErrorLog to evoke the logging
//		behavior in the destinations owned by the ELadminstrator.
//
// ----------------------------------------------------------------------
//
// ELadministrator   The singleton logger class.  One does not instantiate
//		     an ELadministrator.  Instead, do
//			ELadministrator * logger = ELadministrator::instance();
//		     to get a pointer to the (unique) ELadministrator.
//
//	Only the framework should use ELadministrator directly.
//	Physicist users get at it indirectly through using an ErrorLog
//	set up in their Module class.
//
// ELadminDestroyer  A class whose sole purpose is the destruction of the
//                   ELadministrator when the program is over.  Right now,
//                   we do not have anything that needs to be done when the
//                   ELadministrator (and thus the error logger) goes away;
//                   but since by not deleting the copies of ELdestination's
//                   that were attached we would be left with an apparent
//                   memory leak, we include a protected destructor which will
//                   clean up.  ELadminDestroyer provides the only way for
//                   this destructor to be called.
//
// ----------------------------------------------------------------------
//
// 7/2/98 mf	Created file.
// 2/29/00 mf	Added method swapContextSupplier for ELrecv to use.
// 4/5/00 mf	Added method swapProcess for same reason:  ELrecv wants to
//		be able to mock up the process and reset it afterward.
// 6/6/00 web	Consolidate ELadministrator/X; adapt to consolidated
//		ELcout/X.
// 6/14/00 web	Declare classes before granting friendship.
// 6/4/01  mf	Grant friedship to ELtsErrorLog
// 3/6/02  mf   Items for recovering handles to attached destinations:
//		the attachedDestinations map, 
//		an additional signature for attach(), 
//		and getELdestControl() method
// 3/17/04 mf	exitThreshold and setExitThreshold
// 1/10/06 mf	finish
//
// ----------------------------------------------------------------------

#include "FWCore/MessageService/interface/ELdestControl.h"

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELlist.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"

#include "boost/shared_ptr.hpp"

namespace edm {       
namespace service {       


// ----------------------------------------------------------------------
// Prerequisite classes:
// ----------------------------------------------------------------------

class ELdestination;
class ELcout;
class MessageLoggerScribe;


// ----------------------------------------------------------------------
// ELadministrator:
// ----------------------------------------------------------------------

class ELadministrator  {	// *** Destructable Singleton Pattern ***

  friend class MessageLoggerScribe;	// proper ELadministrator cleanup
  friend class ThreadSafeLogMessageLoggerScribe;	// proper ELadministrator cleanup
  friend class ELcout;			// ELcout behavior

// *** Error Logger Functionality ***

public:
  
  ~ELadministrator();


  //Replaces ErrorLog which is no longer needed
  void log(edm::ErrorObj & msg);
  
  // ---  furnish/recall destinations:
  //
  ELdestControl attach( const ELdestination & sink );
  ELdestControl attach( const ELdestination & sink, const ELstring & name );

  // ---  handle severity information:
  //
  ELseverityLevel  checkSeverity();
  int severityCount( const ELseverityLevel & sev ) const;
  int severityCount( const ELseverityLevel & from,
	 	     const ELseverityLevel & to ) const;
  void resetSeverityCount( const ELseverityLevel & sev );
  void resetSeverityCount( const ELseverityLevel & from,
	 	           const ELseverityLevel & to );
  void resetSeverityCount();			// reset all

  // ---  apply the following actions to all attached destinations:
  //
  void setThresholds( const ELseverityLevel & sev );
  void setLimits    ( const ELstring        & id,  int limit    );
  void setLimits    ( const ELseverityLevel & sev, int limit    );
  void setIntervals ( const ELstring        & id,  int interval );
  void setIntervals ( const ELseverityLevel & sev, int interval );
  void setTimespans ( const ELstring        & id,  int seconds  );
  void setTimespans ( const ELseverityLevel & sev, int seconds  );
  void wipe();
  void finish();
  
protected:
  // ---  member data accessors:
  //
  const ELseverityLevel       & abortThreshold() const;
  const ELseverityLevel       &  exitThreshold() const;
  std::list<boost::shared_ptr<ELdestination> >  & sinks();
  const ELseverityLevel       & highSeverity() const;
  int                           severityCounts( int lev ) const;

protected:
  // ---  traditional birth/death, but disallowed to users:
  //
  ELadministrator();

private:

  // ---  traditional member data:
  //
  std::list<boost::shared_ptr<ELdestination> > sinks_;		
  ELseverityLevel            highSeverity_;
  int                        severityCounts_[ ELseverityLevel::nLevels ];

  std::map < ELstring, boost::shared_ptr<ELdestination> > attachedDestinations_;

};  // ELadministrator


// ----------------------------------------------------------------------


}        // end of namespace service
}        // end of namespace edm


#endif  // MessageService_ELadministrator_h
