#ifndef MessageLogger_MessageLogger_h
#define MessageLogger_MessageLogger_h

// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     <none>
// Functions:   LogSystem,   LogError,   LogWarning, LogInfo,     LogDebug
//              LogAbsolute, LogProblem, LogPrint,   LogVerbatim, LogTrace
//

//
// Original Author:  W. Brown and M. Fischler
//         Created:  Fri Nov 11 16:38:19 CST 2005
//     Major Split:  Tue Feb 14 11:00:00 CST 2006
//		     See MessageService/interface/MessageLogger.h
// $Id: MessageLogger.h,v 1.20 2007/02/13 22:39:48 marafino Exp $
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
// =================================================

// system include files

#include <memory>
#include <string>

// user include files

// forward declarations

#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"

namespace edm  {

class LogWarning
{
public:
  explicit LogWarning( std::string const & id ) 
    : ap ( edm::MessageDrop::instance()->warningEnabled ? new MessageSender(ELwarning,id) : 0 )
  { }

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
  
};  // LogWarning

class LogError
{
public:
  explicit LogError( std::string const & id ) 
    : ap( new MessageSender(ELerror,id) )
  { }

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

};  // LogError

class LogSystem
{
public:
  explicit LogSystem( std::string const & id ) 
    : ap( new MessageSender(ELsevere,id) )
  { }

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

};  // LogSystem

class LogInfo
{
public:
  explicit LogInfo( std::string const & id ) 
    : ap ( edm::MessageDrop::instance()->infoEnabled ? new MessageSender(ELinfo,id) : 0 )
  { }

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
  
};  // LogInfo

class LogVerbatim						// change log 2
{
public:
  explicit LogVerbatim( std::string const & id ) 
    : ap( new MessageSender(ELinfo,id,true) ) // the true is the verbatim arg 
  { }

  template< class T >
    LogVerbatim & 
    operator<< (T const & t)  { (*ap) << t; return *this; }
  LogVerbatim & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
    				      { (*ap) << f; return *this; }
  LogVerbatim & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
    				      { (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 
  
};  // LogVerbaitm

class LogPrint						// change log 3
{
public:
  explicit LogPrint( std::string const & id ) 
    : ap( new MessageSender(ELwarning,id,true) ) // the true is the Print arg 
  { }

  template< class T >
    LogPrint & 
    operator<< (T const & t)  { (*ap) << t; return *this; }
  LogPrint & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
    				      { (*ap) << f; return *this; }
  LogPrint & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
    				      { (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 
  
};  // LogPrint

static 
std::string
onlyLowestDirectory(const std::string & file) {
  std::string::size_type lastSlash = file.find_last_of('/');
  if (lastSlash == std::string::npos) return file;
  if (lastSlash == file.size()-1)     return file;
  return file.substr(lastSlash+1, file.size()-lastSlash-1);
}

class LogProblem					// change log 4
{
public:
  explicit LogProblem( std::string const & id ) 
    : ap( new MessageSender(ELerror,id,true) )
  { }

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

};  // LogProblem

class LogAbsolute					// change log 4
{
public:
  explicit LogAbsolute( std::string const & id ) 
    : ap( new MessageSender(ELsevere,id,true) )
  { }

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

};  // LogAbsolute



void LogStatistics(); 

class LogDebug_
{
public:
  explicit LogDebug_( std::string const & id, std::string const & file, int line ) 
    : ap( new MessageSender(ELsuccess,id) )
  { *this << " " << onlyLowestDirectory(file) << ":" << line << "\n"; }
							// change log 1
  template< class T >
    LogDebug_ & 
    operator<< (T const & t)  { (*ap) << t; return *this; }
  LogDebug_ & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
    				      { (*ap) << f; return *this; }
  LogDebug_ & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
    				      { (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 

};  // LogDebug_

class LogTrace_
{
public:
  explicit LogTrace_( std::string const & id ) 
    : ap( new MessageSender(ELsuccess,id,true) )
  {  }
  template< class T >
    LogTrace_ & 
    operator<< (T const & t)  { (*ap) << t; return *this; }
  LogTrace_ & 
  operator<< ( std::ostream&(*f)(std::ostream&))  
    			      { (*ap) << f; return *this; }
  LogTrace_ & 
  operator<< ( std::ios_base&(*f)(std::ios_base&) )  
    			      { (*ap) << f; return *this; }     

private:
  std::auto_ptr<MessageSender> ap; 

};  // LogTrace_

extern LogDebug_ dummyLogDebugObject_;
extern LogTrace_ dummyLogTraceObject_;

class Suppress_LogDebug_ 
{ 
  // With any decent optimization, use of Suppress_LogDebug_ (...)
  // including streaming of items to it via operator<<
  // will produce absolutely no executable code.
public:
  template< class T >
    Suppress_LogDebug_ & 
    operator<< (T const & t)  { return *this; }
};  // Suppress_LogDebug_

  bool isDebugEnabled();
  bool isInfoEnabled();
  bool isWarningEnabled();

}  // namespace edm


// If ML_DEBUG is defined, LogDebug is active.  
// Otherwise, LogDebug is supressed if either ML_NDEBUG or NDEBUG is defined.
#undef EDM_MESSAGELOGGER_SUPPRESS_LOGDEBUG
#ifdef NDEBUG
#define EDM_MESSAGELOGGER_SUPPRESS_LOGDEBUG
#endif
#ifdef ML_NDEBUG
#define EDM_MESSAGELOGGER_SUPPRESS_LOGDEBUG
#endif
#ifdef ML_DEBUG
#undef EDM_MESSAGELOGGER_SUPPRESS_LOGDEBUG
#endif

#ifdef EDM_MESSAGELOGGER_SUPPRESS_LOGDEBUG 
#define LogDebug(id) edm::Suppress_LogDebug_()
#define LogTrace(id) edm::Suppress_LogDebug_()
#else
#define LogDebug(id)                                 \
  ( !edm::MessageDrop::instance()->debugEnabled )    \
    ?  edm::dummyLogDebugObject_                     \
    :  edm::LogDebug_(id, __FILE__, __LINE__)
#define LogTrace(id)                                 \
  ( !edm::MessageDrop::instance()->debugEnabled )    \
    ?  edm::dummyLogTraceObject_                     \
    :  edm::LogTrace_(id)
#endif
#undef EDM_MESSAGELOGGER_SUPPRESS_LOGDEBUG
							// change log 1, 2
#endif  // MessageLogger_MessageLogger_h

