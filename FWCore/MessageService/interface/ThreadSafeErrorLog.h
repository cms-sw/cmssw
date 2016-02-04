#ifndef THREADSAFEERRORLOG_H
#define FWCore_MessageService_ThreadSafeErrorLog_h

// ----------------------------------------------------------------------
//
// ThreadSafeErrorLog 	provides interface to the module-wide variable by which
//			users issue log messages, but utilizes a user-supplied
//			mutex class to do the work in a thread-safe fashion.
//
// 5/29/01 mf	Created file.
//
//	Each thread (or each entitiy that could in principle be building up 
//	an error message) should have its own ThreadSafeErrorLog so that
// 	composition of multiple messages simultaneously will not lead to 
//	jumbled log output.
//
// ----------------------------------------------------------------------



#include "FWCore/MessageService/interface/ELtsErrorLog.h"

namespace edm {       
namespace service {       


// ----------------------------------------------------------------------
// Prerequisite classes:
// ----------------------------------------------------------------------

class ELadministrator;
class ELdestControl;


// ----------------------------------------------------------------------
// ThreadSafeErrorLog:
// ----------------------------------------------------------------------

template <class Mutex>
class ThreadSafeErrorLog  : public ELtsErrorLog {

  // Mutex represents the user-defined locking mechanism, which must
  // work as follows:  Any instance of a Mutex will when constructed
  // obtain the right-to-log-an-error semaphore, and will relinquish 
  // that right when it is destructed.

public:

// ----------------------------------------------------------------------
// -----  Methods for physicists logging errors:
// ----------------------------------------------------------------------

  // -----  start a new logging operation:
  //
  inline ThreadSafeErrorLog & operator()
		( const ELseverityLevel & sev, const ELstring & id );

  inline ErrorLog & operator()( int debugLevel );

  // -----  mutator:
  //
  using ELtsErrorLog::setSubroutine;

  // -----  logging operations:
  //

  inline ThreadSafeErrorLog & emitToken(const ELstring & s); 
  			 		// accumulate one part of a message

  inline ThreadSafeErrorLog & operator()( ErrorObj & msg ); 
				 	// an entire message

  inline ThreadSafeErrorLog & completeMsg();  // no more parts forthcoming

// ----------------------------------------------------------------------
// -----  Methods meant for the Module base class in the framework:
// ----------------------------------------------------------------------

  // -----  birth/death:
  //
  inline ThreadSafeErrorLog();
  inline ThreadSafeErrorLog( const ELstring & pkgName );
  inline ThreadSafeErrorLog( const ErrorLog & ee );
  inline ThreadSafeErrorLog( const ThreadSafeErrorLog<Mutex> & ee );
  inline virtual ~ThreadSafeErrorLog();

  // -----  mutators:
  //
  using ELtsErrorLog::setModule;		// These two are 
  using ELtsErrorLog::setPackage;		//   IDENTICAL

  using ELtsErrorLog::setProcess;		
	// Unlike ErrorLog, ThreadSafeErrorLog can have a process name
	// distinct from that found in the ELadministrator

  // -----  logging collected message:
  //
  inline ThreadSafeErrorLog & operator()( int nbytes, char * data );

  // -----  advanced control options:

  using ELtsErrorLog::setHexTrigger;		
  using ELtsErrorLog::setDiscardThreshold;		
  using ELtsErrorLog::getELdestControl;		
  using ELtsErrorLog::setDebugVerbosity;
  using ELtsErrorLog::setDebugMessages;
		
  // -----  No member data; it is all held by ELtsErrorLog

};  // ThreadSafeErrorLog

// ----------------------------------------------------------------------
// Global functions:
// ----------------------------------------------------------------------

template <class Mutex>
inline ThreadSafeErrorLog<Mutex> & operator<<
	( ThreadSafeErrorLog<Mutex> & e, void (* f)(ErrorLog &) );
				// allow log << endmsg
				// SAME arg. signature as for ErrorLog

template <class Mutex, class T>
inline ThreadSafeErrorLog<Mutex> &
        operator<<( ThreadSafeErrorLog<Mutex> & e, const T & t );

// ----------------------------------------------------------------------


}        // end of namespace service
}       // end of namespace edm


// ----------------------------------------------------------------------
// .icc
// ----------------------------------------------------------------------

#define THREADSAFEERRORLOG_ICC
  #include "FWCore/MessageLogger/interface/ThreadSafeErrorLog.icc"
#undef  THREADSAFEERRORLOG_ICC


// ----------------------------------------------------------------------


#endif  // THREADSAFEERRORLOG_H
