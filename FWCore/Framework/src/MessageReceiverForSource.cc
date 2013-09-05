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
#include <cerrno>
#include <cstring>

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
MessageReceiverForSource::MessageReceiverForSource(int parentSocket, int parentPipe) :
m_parentSocket(parentSocket),
m_parentPipe(parentPipe),
m_maxFd(parentPipe),
m_startIndex(0),
m_numberOfConsecutiveIndices(0),
m_numberToSkip(0)
{
   if (parentSocket > parentPipe) {
      m_maxFd = parentSocket;
   }
   m_maxFd++;
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
/*
 * The child side of the parent-child communication.  See MessageSenderForSource in 
 * EventProcessor.cc for more information.
 *
 * If the parent terminates before/during send, the send will immediately fail.
 * If the parent terminates after send and before recv is successful, the recv may hang.
 * Hence, this socket has been marked as non-blocking and we wait until select indicate
 * data is available to read.  If the select times out, we write a byte on the watchdog
 * pipe.  The byte is meaningless - we are just testing to see if there is an error
 * (as EPIPE will be returned if the parent has exited).
 *
 * Note: if the parent dies between send/recv, it may take the child up to a second to
 * notice.  This was the best I could do without adding another watchdog pipe.
 */

void 
MessageReceiverForSource::receive()
{
   unsigned long previousStartIndex = m_startIndex;
   unsigned long previousConsecutiveIndices = m_numberOfConsecutiveIndices;
  
   //request more work from the parent
   ssize_t rc, rc2;

   {
      MessageForParent parentMessage;
      errno = 0;

      // If parent has died, this will fail with "connection refused"
      if ((rc = send(m_parentSocket, reinterpret_cast<char *>(&parentMessage), parentMessage.sizeForBuffer(), 0)) != static_cast<int>(parentMessage.sizeForBuffer())) {
         m_numberOfConsecutiveIndices=0;
         m_startIndex=0;
         if (rc == -1) {
            throw cms::Exception("MulticoreCommunicationFailure") << "failed to send data to controller: errno=" << errno << " : " << strerror(errno);
         }
         throw cms::Exception("MulticoreCommunicationFailure") << "Unable to write full message to controller (" << rc << " of " << parentMessage.sizeForBuffer() << " byte written)";
      }
   }

   {  
      MessageForSource message;
      fd_set readSockets, errorSockets;
      errno = 0;

      do {
         // We reset the FD set after each select, as select changes the sets we pass it.
         FD_ZERO(&errorSockets); FD_SET(m_parentPipe, &errorSockets); FD_SET(m_parentSocket, &errorSockets);
         FD_ZERO(&readSockets); FD_SET(m_parentSocket, &readSockets);
         struct timeval tv; tv.tv_sec = 1; tv.tv_usec = 0;
         while (((rc = select(m_maxFd, &readSockets, NULL, &errorSockets, &tv)) < 0) && (errno == EINTR)) {}

         if (rc == 0) {
             // If we timeout waiting for the parent, this will detect if the parent is still alive.
             while (((rc2 = write(m_parentPipe, "\0", 1)) < 0) && (errno == EINTR)) {}
             if (rc2 < 0) {
                if (errno == EPIPE) {
                   throw cms::Exception("MulticoreCommunicationFailure") << "Parent process appears to be dead.";
                }
                throw cms::Exception("MulticoreCommunicationFailure") << "Cannot write to parent:  errno=" << errno << " : " << strerror(errno);
             }
         }
      } while (rc == 0);

      // Check for errors
      if (FD_ISSET(m_parentSocket, &errorSockets) || FD_ISSET(m_parentPipe, &errorSockets)) {
         throw cms::Exception("MulticoreCommunicationFailure") << "Cannot communicate with parent (fd=" << m_parentSocket << "): errno=" << errno << " : " << strerror(errno);
      }

      if (!FD_ISSET(m_parentSocket, &readSockets)) {
         throw cms::Exception("MulticoreCommunicationFailure") << "Unable to read from parent socket";
      }

      // Note the parent can die between the send and recv; in this case, the recv will hang
      // forever.  Thus, this socket has been marked as non-blocking.  According to man pages, it's possible
      // for no data to be recieved after select indicates it's ready.  The size check below will catch this,
      // but not recover from it.  The various edge cases seemed esoteric enough to not care.
      rc = recv(m_parentSocket, &message, MessageForSource::sizeForBuffer(), 0);
      if (rc < 0) {
         m_numberOfConsecutiveIndices=0;
         m_startIndex=0;
         throw cms::Exception("MulticoreCommunicationFailure")<<"failed to receive data from controller: errno="<<errno<<" : "<<strerror(errno);
      }

      if (rc != (int)MessageForSource::sizeForBuffer()) {
         m_numberOfConsecutiveIndices=0;
         m_startIndex=0;
         throw cms::Exception("MulticoreCommunicationFailure")<<"Incorrect number of bytes received from controller (got " << rc << ", expected " << MessageForSource::sizeForBuffer() << ")";
      }

      m_startIndex = message.startIndex;
      m_numberOfConsecutiveIndices = message.nIndices;
      m_numberToSkip = m_startIndex-previousStartIndex-previousConsecutiveIndices;

      //printf("Start index: %lu, number consecutive: %lu, number to skip: %lu\n", m_startIndex, m_numberOfConsecutiveIndices, m_numberToSkip);
   }

   return;
}

//
// const member functions
//

//
// static member functions
//
