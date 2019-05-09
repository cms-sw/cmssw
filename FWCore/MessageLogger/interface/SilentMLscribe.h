#ifndef FWCore_MessageLogger_SilentMLscribe_h
#define FWCore_MessageLogger_SilentMLscribe_h
// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     SilentMLscribe
//
/**\class SilentMLscribe SilentMLscribe.h FWCore/MessageLogger/interface/SilentMLscribe.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jul 30 09:57:52 CDT 2009
//

// system include files

// user include files
#include "FWCore/MessageLogger/interface/AbstractMLscribe.h"

// forward declarations
namespace edm {
  namespace service {
    class SilentMLscribe : public AbstractMLscribe {
    public:
      SilentMLscribe();
      ~SilentMLscribe() override;

      // ---------- member functions ---------------------------

      void runCommand(MessageLoggerQ::OpCode opcode, void* operand) override;

    private:
      SilentMLscribe(const SilentMLscribe&) = delete;  // stop default

      const SilentMLscribe& operator=(const SilentMLscribe&) = delete;  // stop default

      // ---------- member data --------------------------------
    };
  }  // namespace service
}  // namespace edm

#endif
