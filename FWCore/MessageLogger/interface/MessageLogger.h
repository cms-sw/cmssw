#ifndef MessageLogger_MessageLogger_h
#define MessageLogger_MessageLogger_h

// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     <none>
// Functions:   LogError, LogWarning, LogInfo,     LogDebug
//                                  , LogVerbatim, LogTrace
//

//
// Original Author:  W. Brown and M. Fischler
//         Created:  Fri Nov 11 16:38:19 CST 2005
//     Major Split:  Tue Feb 14 11:00:00 CST 2006
//		     See MessageService/interface/MessageLogger.h
// $Id: MessageLogger.h,v 1.16 2006/08/18 16:28:43 marafino Exp $
//
// =================================================
// Change log
//
// 1 mf 5/11/06	    Added a space before the file/line string in LogDebug_
//		    to avoid the run-together with the run and event number
//
// 2 mf 6/6/06	    Added LogVerbatim and LogTrace
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
  
};  // LogInfo

static 
std::string
onlyLowestDirectory(const std::string & file) {
  std::string::size_type lastSlash = file.find_last_of('/');
  if (lastSlash == std::string::npos) return file;
  if (lastSlash == file.size()-1)     return file;
  return file.substr(lastSlash+1, file.size()-lastSlash-1);
}

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
#define LogDebug(id) edm::Suppress_LogDebug_();
#define LogTrace(id) edm::Suppress_LogDebug_();
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

