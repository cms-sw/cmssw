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
#include <errno.h>
#include <string.h>

// user include files
#include "FWCore/Framework/interface/MessageReceiverForSource.h"
#include "FWCore/Utilities/interface/Exception.h"

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
MessageReceiverForSource::MessageReceiverForSource(int iQueueID):
m_queueID(iQueueID),
m_startIndex(0),
m_numberOfConsecutiveIndices(0),
m_numberToSkip(0)
{
}
/*
MessageReceiverForSource::MessageReceiverForSource(unsigned int iChildIndex, unsigned int iNumberOfChildren, unsigned int iNumberOfSequentialEvents):
m_startIndex(0),
m_numberOfConsecutiveIndices(0),
m_numberToSkip(0),
m_forkedChildIndex(iChildIndex),
m_numberOfIndicesToSkip(iNumberOfSequentialEvents*(iNumberOfChildren-1)),
m_originalConsecutiveIndices(iNumberOfSequentialEvents)
{
}
 */

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
   /*
   //DUMMY
   if (m_originalConsecutiveIndices != m_numberOfConsecutiveIndices) {
      m_numberOfConsecutiveIndices = m_originalConsecutiveIndices;
      m_startIndex = m_numberOfConsecutiveIndices*m_forkedChildIndex;
   } else {
      m_startIndex += m_numberOfConsecutiveIndices+m_numberOfIndicesToSkip;
   }*/
   MessageForSource message;
   errno = 0;
   int value = msgrcv(m_queueID, &message, MessageForSource::sizeForBuffer(), MessageForSource::messageType(), 0);
   if (value < 0) {
      m_numberOfConsecutiveIndices=0;
      throw cms::Exception("MulticoreCommunicationFailure")<<"failed to receive data from controller: errno="<<errno<<" : "<<strerror(errno);
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
