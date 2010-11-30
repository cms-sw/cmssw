#ifndef MessageLogger_MessageLogger_h
#define MessageLogger_MessageLogger_h

// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     <none>
// Functions:   LogSystem,   LogError,   LogWarning, LogInfo,     LogDebug
//              LogAbsolute, LogProblem, LogPrint,   LogVerbatim, LogTrace
//			     LogImportant
//

//
// Original Author:  W. Brown and M. Fischler
//         Created:  Fri Nov 11 16:38:19 CST 2005
//     Major Split:  Tue Feb 14 11:00:00 CST 2006
//		     See MessageService/interface/MessageLogger.h
//
// =================================================
// Change log
//
// 1 mf 5/11/06	    Added a space before the file/line string in LogDebug_
//		    to avoid the run-together with the run and event number
//
// 2 mf 6/6/06	    Added LogVerbatim and LogTrace
//
// 3 mf 10/30/06    Added LogSystem and LogPrint
//
// 4 mf 6/1/07      Added LogAbsolute and LogProblem
//
// 5 mf 7/24/07     Added HaltMessageLogging
//
// 6 mf 8/7/07      Added FlushMessageLog
//
// 7 mf 8/7/07      Added GroupLogStatistics(category)
//
// 8 mf 12/12/07    Reworked LogDebug macro, LogDebug_class,, and similarly
//		    for LogTrace, to avoid the need for the static dummy
//		    objects.  This cures the use-of-thread-commands-after-
//		    exit problem in programs that link but do not use the
//		    MessageLogger.
// 
// 9  mf 12/12/07   Check for subtly terrible situation of copying and then
//		    writing to a LogDebug_ object.  Also forbid copying any
//		    of the ordinary LogXXX objects (since that implies either
//		    copying a MessageSender, or having a stale copy available
//		    to lose message contents).
//
// 10 mf 12/14/07   Moved the static free function onlyLowestDirectory
//		    to a class member function of LogDebug_, changing
//		    name to a more descriptive stripLeadingDirectoryTree.
//		    Cures the 2600-copies-of-this-function complaint.
//
// 11 mf  6/24/08   Added LogImportant which is LogProblem.  For output
//		    which "obviously" ought to emerge, but allowing specific
//		    suppression as if it were at the LogError level, rather
//		    than the non-suppressible LogAbsolute.
//
// 12 ge  9/12/08   MessageLogger now works even when compiled with -DNDEBUG.
//                  The problem was that Suppress_LogDebug_ was missing the operator<<
//                  needed for streaming `std::iomanip`s.
//
// 13 wmtan 11/18/08 Use explicit non-inlined destructors
//
// 14 mf  3/23/09   ap.get() used whenever possible suppression, to avoid
//		    null pointer usage
//
// 15 mf  8/11/09   provision for control of standalone threshold and ignores
//
// 16 mf  10/2/09  Correct mission in logVerbatim and others of check for
//		   whether this severity is enabled
//
// 17 wmtan 10/29/09 Out of line LogDebug_ and LogTrace_ constructors.
//
// 18 wmtan 07/08/10 Remove unnecessary includes
//
// 19 mf  09/21/10 !!! BEHAVIOR CHANGE: LogDebug suppression.
//                 The sole preprocessor symbol controlling suppression of 
//                 LogDebug is EDM_ML_DEBUG.  If EDM_ML_DEBUG is not defined
//                 then LogDebug is suppressed.  Thus by default LogDebug is 
//                 suppressed.
//
// 20 mf  09/21/10 The mechanism of LogDebug is modified such that if LogDebug
//                 is suppressed, whether via lack of the EDM_ML_DEBUG symbol 
//                 or dynabically via !debgEnabled, all code past the 
//                 LogDebug(...), including operator<< and functions to 
//                 prepare the output strings, is squelched.  This was the
//                 original intended behavior.  
//
//  ************   Note that in this regard, LogDebug behaves like assert:
//                 The use of functions having side effects, on a LogDebug
//                 statement (just as within an assert()), is unwise as it
//                 makes turning on the debugging aid alter the program
//                 behavior.
//
// 21  mf 9/23/10  Support for situations where no thresholds are low
//                 enough to react to LogDebug (or info, or warning).
//		   A key observation is that while debugEnabled
//		   should in principle be obtained via an instance() of 
//		   MessageDrop, debugAlwaysSuppressed is universal across
//		   threads and hence is properly just a class static, which
//		   is much quicker to check.
//
// 22  mf 9/27/10  edmmltest::LogWarningThatSuppressesLikeLogInfo, 
//		   a class provided solely to allow testing of the feature
//		   that if all destinations have threshold too high, then
//		   a level of messages (in this case, INFO) will be suppressed
//		   without even being seen by the destinations. 
//
// 23 mf 11/30/10  SnapshotMessageLog() method to force MessageDrop to 
//		   capture any pointed-to strings in anticipation of key 
//		   objects going away before a message is going to be issued.
//
// =================================================

// system include files

#include <memory>
#include <string>

// user include files

// forward declarations

#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/Utilities/interface/EDMException.h"		// Change log 8


namespace edm  {

class LogWarning
{
public:
  explicit LogWarning( std::string const & id ) 
    : ap ( (!MessageDrop::warningAlwaysSuppressed 		// Change log 21
            && edm::MessageDrop::instance()->warningEnabled) ?
           new MessageSender(ELwarning,id) : 0 )
  { }
  ~LogWarning();						// Change log 13

  template< class T >
    LogWarning & 
    operator<< (T const & t)  { if(ap.get()) (*ap) << t; return *this; }
  LogWarning & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				      { if(ap.get()) (*ap) << f; return *this; }
  LogWarning & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				      { if(ap.get()) (*ap) << f; return *this; }     
private:
  std::auto_ptr<MessageSender> ap; 
  LogWarning( LogWarning const& );				// Change log 9
   
};  // LogWarning

class LogError
{
public:
  explicit LogError( std::string const & id ) 
    : ap( new MessageSender(ELerror,id) )
  { }
  ~LogError();							// Change log 13

  template< class T >
    LogError & 
    operator<< (T const & t)  { (*ap) << t; return *this; }
  LogError & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				      { (*ap) << f; return *this; }
  LogError & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				      { (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 
  LogError( LogError const& );					// Change log 9

};  // LogError

class LogSystem
{
public:
  explicit LogSystem( std::string const & id ) 
    : ap( new MessageSender(ELsevere,id) )
  { }
  ~LogSystem();							// Change log 13

  template< class T >
    LogSystem & 
    operator<< (T const & t)  { (*ap) << t; return *this; }
  LogSystem & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				      { (*ap) << f; return *this; }
  LogSystem & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				      { (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 
  LogSystem( LogSystem const& );				// Change log 9

};  // LogSystem

class LogInfo				
{
public:
  explicit LogInfo( std::string const & id ) 
    : ap ( (!MessageDrop::infoAlwaysSuppressed 		// Change log 21
            && edm::MessageDrop::instance()->infoEnabled) ?
           new MessageSender(ELinfo,id) : 0 )
  { }
  ~LogInfo();							// Change log 13

  template< class T >
    LogInfo & 
    operator<< (T const & t)  { if(ap.get()) (*ap) << t; return *this; }
  LogInfo & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				      { if(ap.get()) (*ap) << f; return *this; }
  LogInfo & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				      { if(ap.get()) (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 
  LogInfo( LogInfo const& );					// Change log 9
  
};  // LogInfo

// verbatim version of LogInfo
class LogVerbatim						// change log 2
{
public:
  explicit LogVerbatim( std::string const & id ) 
    : ap ( (!MessageDrop::infoAlwaysSuppressed 		// Change log 21
            && edm::MessageDrop::instance()->infoEnabled) ?
           new MessageSender(ELinfo,id,true) : 0 ) // true for verbatim
  { }
  ~LogVerbatim();						// Change log 13

  template< class T >
    LogVerbatim & 
    operator<< (T const & t)  { if(ap.get()) (*ap) << t; return *this; } 
								// Change log 14
  LogVerbatim & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				      { if(ap.get()) (*ap) << f; return *this; }
  LogVerbatim & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				      { if(ap.get()) (*ap) << f; return *this; }   

private:
  std::auto_ptr<MessageSender> ap; 
  LogVerbatim( LogVerbatim const& );				// Change log 9
  
};  // LogVerbatim

// verbatim version of LogWarning
class LogPrint							// change log 3
{
public:
  explicit LogPrint( std::string const & id ) 
    : ap ( (!MessageDrop::warningAlwaysSuppressed 		// Change log 21
            && edm::MessageDrop::instance()->warningEnabled) ?
           new MessageSender(ELwarning,id,true) : 0 ) // true for verbatim
  { }
  ~LogPrint();							// Change log 13

  template< class T >
    LogPrint & 
    operator<< (T const & t)  { if(ap.get()) (*ap) << t; return *this; } 
								// Change log 14
  LogPrint & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				{ if(ap.get()) (*ap) << f; return *this; }
  LogPrint & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				{ if(ap.get()) (*ap) << f; return *this; }      

private:
  std::auto_ptr<MessageSender> ap; 
  LogPrint( LogPrint const& );					// Change log 9
  
};  // LogPrint


// verbatim version of LogError
class LogProblem						// change log 4
{
public:
  explicit LogProblem( std::string const & id ) 
    : ap( new MessageSender(ELerror,id,true) )
  { }
  ~LogProblem();						// Change log 13

  template< class T >
    LogProblem & 
    operator<< (T const & t)  { (*ap) << t; return *this; }
  LogProblem & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				      { (*ap) << f; return *this; }
  LogProblem & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				      { (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 
  LogProblem( LogProblem const& );				// Change log 9

};  // LogProblem

// less judgemental verbatim version of LogError
class LogImportant						// change log 11
{
public:
  explicit LogImportant( std::string const & id ) 
    : ap( new MessageSender(ELerror,id,true) )
  { }
  ~LogImportant();						 // Change log 13

  template< class T >
    LogImportant & 
    operator<< (T const & t)  { (*ap) << t; return *this; }
  LogImportant & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				      { (*ap) << f; return *this; }
  LogImportant & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				      { (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 
  LogImportant( LogImportant const& );				// Change log 9

};  // LogImportant

// verbatim version of LogSystem
class LogAbsolute						// change log 4
{
public:
  explicit LogAbsolute( std::string const & id ) 
    : ap( new MessageSender(ELsevere,id,true) )
  { }
  ~LogAbsolute();						// Change log 13

  template< class T >
    LogAbsolute & 
    operator<< (T const & t)  { (*ap) << t; return *this; }
  LogAbsolute & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				      { (*ap) << f; return *this; }
  LogAbsolute & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				      { (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 
  LogAbsolute( LogAbsolute const& );				// Change log 9

};  // LogAbsolute

std::string stripLeadingDirectoryTree(const std::string & file);

// change log 10:  removed onlyLowestDirectory()

void LogStatistics(); 

class LogDebug_
{
public:
  explicit LogDebug_( std::string const & id, std::string const & file, int line ); // Change log 17
  explicit LogDebug_()  : ap(), debugEnabled(false) {}		// Change log 8	
  ~LogDebug_(); 

  template< class T >
    LogDebug_ & 
    operator<< (T const & t)  
    { if (!debugEnabled) return *this;				// Change log 8
      if (ap.get()) (*ap) << t; 
      else Exception::throwThis
       (edm::errors::LogicError,"operator << to stale copied LogDebug_ object"); 
      return *this; }
  LogDebug_ & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
    { if (!debugEnabled) return *this;				// Change log 8
      if (ap.get()) (*ap) << f; 
      else Exception::throwThis
       (edm::errors::LogicError,"operator << to stale copied LogDebug_ object"); 
      return *this; }
  LogDebug_ & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
    { if (!debugEnabled) return *this;				// Change log 8
      if (ap.get()) (*ap) << f; 
      else Exception::throwThis
       (edm::errors::LogicError,"operator << to stale copied LogDebug_ object"); 
      return *this; }
			   // Change log 8:  The tests for ap.get() being null 

private:
  std::auto_ptr<MessageSender> ap; 
  bool debugEnabled;
  std::string stripLeadingDirectoryTree (const std::string & file) const;
								// change log 10
};  // LogDebug_

class LogTrace_
{
public:
  explicit LogTrace_( std::string const & id );			// Change log 13
  explicit LogTrace_()  : ap(), debugEnabled(false) {}		// Change log 8	
  ~LogTrace_(); 

  template< class T >
    LogTrace_ & 
    operator<< (T const & t)  
    { if (!debugEnabled) return *this;				// Change log 8
      if (ap.get()) (*ap) << t; 
      else Exception::throwThis
       (edm::errors::LogicError,"operator << to stale copied LogTrace_ object"); 
      return *this; }
  LogTrace_ & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
    { if (!debugEnabled) return *this;				// Change log 8
      if (ap.get()) (*ap) << f; 
      else Exception::throwThis
       (edm::errors::LogicError,"operator << to stale copied LogTrace_ object"); 
      return *this; }
  LogTrace_ & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
    { if (!debugEnabled) return *this;				// Change log 8
      if (ap.get()) (*ap) << f; 
      else Exception::throwThis
       (edm::errors::LogicError,"operator << to stale copied LogTrace_ object"); 
      return *this; }
			   // Change log 8:  The tests for ap.get() being null 
 
private:
  std::auto_ptr<MessageSender> ap; 
  bool debugEnabled;
  
};  // LogTrace_

// Change log 22
namespace edmmltest {
class LogWarningThatSuppressesLikeLogInfo
{
public:
  explicit LogWarningThatSuppressesLikeLogInfo( std::string const & id ) 
    : ap ( (!MessageDrop::infoAlwaysSuppressed 			// Change log 22
            && edm::MessageDrop::instance()->warningEnabled) ?
           new MessageSender(ELwarning,id) : 0 )
  { }
  ~LogWarningThatSuppressesLikeLogInfo();						
  template< class T >
    LogWarningThatSuppressesLikeLogInfo & 
    operator<< (T const & t)  { if(ap.get()) (*ap) << t; return *this; }
  LogWarningThatSuppressesLikeLogInfo & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
				      { if(ap.get()) (*ap) << f; return *this; }
  LogWarningThatSuppressesLikeLogInfo & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
				      { if(ap.get()) (*ap) << f; return *this; }     
private:
  std::auto_ptr<MessageSender> ap; 
  LogWarningThatSuppressesLikeLogInfo( LogWarningThatSuppressesLikeLogInfo const& );				// Change log 9
   
};  // LogWarningThatSuppressesLikeLogInfo
} // end namespace edmmltest

extern LogDebug_ dummyLogDebugObject_;
extern LogTrace_ dummyLogTraceObject_;

class Suppress_LogDebug_ 
{ 
  // With any decent optimization, use of Suppress_LogDebug_ (...)
  // including streaming of items to it via operator<<
  // will produce absolutely no executable code.
public:
  template< class T >
    Suppress_LogDebug_ &operator<< (T const & t) { return *this; }	// Change log 12
    Suppress_LogDebug_ &operator<< (std::ostream&(*)(std::ostream&)) { return *this; }	// Change log 12
    Suppress_LogDebug_ &operator<< (std::ios_base&(*)(std::ios_base&)) { return *this; } // Change log 12
};  // Suppress_LogDebug_

  bool isDebugEnabled();
  bool isInfoEnabled();
  bool isWarningEnabled();
  void HaltMessageLogging();
  void FlushMessageLog();
  void snapshotMessageLog(); 
  void GroupLogStatistics(std::string const & category);
  bool isMessageProcessingSetUp();

  // Change Log 15
  // The following two methods have no effect except in stand-alone apps
  // that do not create a MessageServicePresence:
  void setStandAloneMessageThreshold    (std::string const & severity);
  void squelchStandAloneMessageCategory (std::string const & category);
  
}  // namespace edm


// change log 19 and change log 20
// The preprocessor symbol controlling suppression of LogDebug is EDM_ML_DEBUG.  Thus by default (BEHAVIOR CHANGE) LogDebug is 
// If LogDebug is suppressed, all code past the LogDebug(...) is squelched.
// See doc/suppression.txt.

#ifndef EDM_ML_DEBUG 
#define LogDebug(id) true ? edm::Suppress_LogDebug_() : edm::Suppress_LogDebug_() 
#define LogTrace(id) true ? edm::Suppress_LogDebug_() : edm::Suppress_LogDebug_() 
#else
// change log 21
#define LogDebug(id) (edm::MessageDrop::debugAlwaysSuppressed || !edm::MessageDrop::debugEnabled) ? edm::LogDebug_() : edm::LogDebug_(id, __FILE__, __LINE__) 
#define LogTrace(id) (edm::MessageDrop::debugAlwaysSuppressed || !edm::MessageDrop::debugEnabled) ? edm::LogTrace_() : edm::LogTrace_(id) 
#endif

#endif  // MessageLogger_MessageLogger_h

