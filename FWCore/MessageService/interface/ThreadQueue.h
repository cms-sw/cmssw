#ifndef FWCore_MessageService_ThreadQueue_h
#define FWCore_MessageService_ThreadQueue_h
// -*- C++ -*-
//
// Package:     MessageService
// Class  :     ThreadQueue
// 
/**\class ThreadQueue ThreadQueue.h FWCore/MessageService/interface/ThreadQueue.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  mf and cdj
//         Created:  Fri Aug  7 10:19:58 CDT 2009
// $Id$
//

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/Utilities/interface/SingleConsumerQ.h"




namespace edm {
namespace service {

class ThreadQueue
{

   public:
      ThreadQueue();
      virtual ~ThreadQueue();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

  // ---  obtain a message from the queue:
  void  consume( MessageLoggerQ::OpCode & opcode, void * & operand );

  // ---  place a message onto the queue:
  void  produce( MessageLoggerQ::OpCode opcode, void *   operand );

 
   private:
      ThreadQueue(const ThreadQueue&); // stop default

      const ThreadQueue& operator=(const ThreadQueue&); // stop default

      // ---------- member data --------------------------------

  // --- buffer parameters:  (were private but needed by MainTrhreadMLscribe
  static  const int  buf_depth = 500;
  static  const int  buf_size  = sizeof(MessageLoggerQ::OpCode)
                               + sizeof(void *);
  SingleConsumerQ  m_buf;

};

} // end namespace service
} // end namespace edm

#endif
