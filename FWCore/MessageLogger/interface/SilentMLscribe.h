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
// $Id: SilentMLscribe.h,v 1.1 2009/07/30 15:33:09 chrjones Exp $
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
         virtual ~SilentMLscribe();
         
         
         // ---------- member functions ---------------------------
         virtual
         void  runCommand(MessageLoggerQ::OpCode  opcode, void * operand);
         
      private:
         SilentMLscribe(const SilentMLscribe&); // stop default
         
         const SilentMLscribe& operator=(const SilentMLscribe&); // stop default
         
         // ---------- member data --------------------------------
         
      };      
   }
}

#endif
