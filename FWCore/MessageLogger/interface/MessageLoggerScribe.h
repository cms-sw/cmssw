#ifndef FWCore_MessageLogger_MessageLoggerScribe_h
#define FWCore_MessageLogger_MessageLoggerScribe_h

#include "FWCore/MessageLogger/interface/ELadministrator.h"
#include "FWCore/MessageLogger/interface/ELdestControl.h"
#include "FWCore/MessageLogger/interface/ErrorLog.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace edm
{


class MessageLoggerScribe
{
public:
  // ---  birth/death:
  MessageLoggerScribe();
  ~MessageLoggerScribe();

  // --- receive and act on messages:
  void  run();

private:
  // --- handle details of configuring via a ParameterSet:
  void  configure_errorlog( ParameterSet const * );

  // --- data:
  ELadministrator *  admin_p;
  ELdestControl      early_dest;
  ErrorLog        *  errorlog_p;

};  // MessageLoggerScribe


}  // namespace edm


#endif  // FWCore_MessageLogger_MessageLoggerScribe_h
