#ifndef FWCore_MessageService_MessageLogger_h
#define FWCore_MessageService_MessageLogger_h

// -*- C++ -*-
//
// Package:     MessageService
// Class  :     MessageLogger
//
/**\class edm::MessageLogger MessageLogger.h FWCore/MessageService/plugins/MessageLogger.h

 Description: Abstract interface for MessageLogger Service

 Usage:
    <usage>

*/
//

// system include files

// user include files

// forward declarations

namespace edm {
  class ModuleCallingContext;

  class MessageLogger {
  public:
    virtual ~MessageLogger();

    virtual void setThreadContext(ModuleCallingContext const&) = 0;

  protected:
    MessageLogger() = default;

  };  // MessageLogger

}  // namespace edm

#endif  // FWCore_MessageService_MessageLogger_h
