#ifndef Services_MessageLogger_h
#define Services_MessageLogger_h

// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageLogger
//
/**\class MessageLogger MessageLogger.h FWCore/Services/interface/MessageLogger.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  W. Brown and M. Fischler
//         Created:  Fri Nov 11 16:38:19 CST 2005
// $Id: MessageLogger.h,v 1.6 2006/01/19 17:12:58 fischler Exp $
//

// system include files

#include <memory>
#include <string>
#include <set>

// user include files

// forward declarations

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "DataFormats/Common/interface/EventID.h"

namespace edm  {
namespace service  {


class MessageLogger
{
public:
  MessageLogger( ParameterSet const &, ActivityRegistry & );

  void  postBeginJob();
  void  postEndJob();

  void  preEventProcessing ( edm::EventID const &, edm::Timestamp const & );
  void  postEventProcessing( Event const &, EventSetup const & );

  void  preModule ( ModuleDescription const & );
  void  postModule( ModuleDescription const & );

  void  fillErrorObj(edm::ErrorObj& obj) const;
  bool  debugEnabled() const { return debugEnabled_; }

  static 
  bool  anyDebugEnabled() { return anyDebugEnabled_; }
  
private:
  // put an ErrorLog object here, and maybe more

  edm::EventID curr_event_;
  std::string curr_module_;

  std::set<std::string> debugEnabledModules_;
  bool debugEnabled_;
  static bool   anyDebugEnabled_;
  static bool everyDebugEnabled_;

};  // MessageLogger


}  // namespace service


class LogWarning
{
public:
  explicit LogWarning( ELstring const & id ) 
    : ap( new MessageSender(ELwarning,id) )
  { }

  template< class T >
    LogWarning & 
    operator<< (T const & t)  { (*ap) << t; return *this; }

private:
  std::auto_ptr<MessageSender> ap; 
  
};  // LogWarning


class LogError
{
public:
  explicit LogError( ELstring const & id ) 
    : ap( new MessageSender(ELerror,id) )
  { }

  template< class T >
    LogError & 
    operator<< (T const & t)  { (*ap) << t; return *this; }

private:
  std::auto_ptr<MessageSender> ap; 

};  // LogError


class LogInfo
{
public:
  explicit LogInfo( ELstring const & id ) 
    : ap( new MessageSender(ELinfo,id) )
  { }

  template< class T >
    LogInfo & 
    operator<< (T const & t)  { (*ap) << t; return *this; }

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
  explicit LogDebug_( ELstring const & id, std::string const & file, int line ) 
    : ap( new MessageSender(ELsuccess,id) )
  { *this << onlyLowestDirectory(file) << ':' << line << ' '; }

  template< class T >
    LogDebug_ & 
    operator<< (T const & t)  { (*ap) << t; return *this; }

private:
  std::auto_ptr<MessageSender> ap; 

};  // LogDebug_

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
#else
#define LogDebug(id)                                                   \
  if ( edm::service::MessageLogger::anyDebugEnabled () &&               \
       edm::Service<edm::service::MessageLogger>()->debugEnabled () )    \
          edm::LogDebug_(id, __FILE__, __LINE__)                               
#endif
#undef EDM_MESSAGELOGGER_SUPPRESS_LOGDEBUG

#endif  // Services_MessageLogger_h

