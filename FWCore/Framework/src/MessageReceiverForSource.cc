// -*- C++ -*-
//
// Package:     Framework
// Class  :     MessageReceiverForSource
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu Dec 30 10:09:50 CST 2010
//

// system include files
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/socket.h>
#include <errno.h>
#include <string.h>

// user include files
#include "FWCore/Framework/interface/MessageReceiverForSource.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "MessageForSource.h"
#include "MessageForParent.h"

using namespace edm::multicore;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MessageReceiverForSource::MessageReceiverForSource(int parentSocket):
m_parentSocket(parentSocket),
m_startIndex(0),
m_numberOfConsecutiveIndices(0),
m_numberToSkip(0)
{
}

// MessageReceiverForSource::MessageReceiverForSource(const MessageReceiverForSource& rhs)
// {
//    // do actual copying here;
// }

//MessageReceiverForSource::~MessageReceiverForSource()
//{
//}

//
// assignment operators
//
// const MessageReceiverForSource& MessageReceiverForSource::operator=(const MessageReceiverForSource& rhs)
// {
//   //An exception safe implementation is
//   MessageReceiverForSource temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
MessageReceiverForSource::receive()
{
   unsigned long previousStartIndex = m_startIndex;
   unsigned long previousConsecutiveIndices = m_numberOfConsecutiveIndices;
  
   //request more work from the parent
   ssize_t rc;
   MessageForParent parentMessage;
   if ((rc = send(m_parentSocket, reinterpret_cast<char *>(&parentMessage), parentMessage.sizeForBuffer(), 0)) != static_cast<int>(parentMessage.sizeForBuffer())) {
     m_numberOfConsecutiveIndices=0;
     m_startIndex=0;
     if (rc == -1) {
       throw cms::Exception("MulticoreCommunicationFailure") << "failed to send data to controller: errno=" << errno << " : " << strerror(errno);
     }
     throw cms::Exception("MulticoreCommunicationFailure") << "Unable to write full message to controller (" << rc << " of " << parentMessage.sizeForBuffer() << " byte written)";
   }
  
   MessageForSource message;
   errno = 0;
   rc = recv(m_parentSocket, &message, MessageForSource::sizeForBuffer(), 0);
   if (rc < 0) {
     m_numberOfConsecutiveIndices=0;
     m_startIndex=0;
     throw cms::Exception("MulticoreCommunicationFailure")<<"failed to receive data from controller: errno="<<errno<<" : "<<strerror(errno);
   }

   /*int value = msgrcv(m_queueID, &message, MessageForSource::sizeForBuffer(), MessageForSource::messageType(), 0);
   if (value < 0) {
      m_numberOfConsecutiveIndices=0;
      throw cms::Exception("MulticoreCommunicationFailure")<<"failed to receive data from controller: errno="<<errno<<" : "<<strerror(errno);
   }
    */
  
   if (rc != (int)MessageForSource::sizeForBuffer()) {
      m_numberOfConsecutiveIndices=0;
      m_startIndex=0;
      throw cms::Exception("MulticoreCommunicationFailure")<<"Incorrect number of bytes received from controller (got " << rc << ", expected " << MessageForSource::sizeForBuffer() << ")";
   }

   m_startIndex = message.startIndex;
   m_numberOfConsecutiveIndices = message.nIndices;   
   m_numberToSkip = m_startIndex-previousStartIndex-previousConsecutiveIndices;
   return;
}

//
// const member functions
//

//
// static member functions
//
