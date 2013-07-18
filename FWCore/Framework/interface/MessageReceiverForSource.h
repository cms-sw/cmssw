#ifndef FWCore_Framework_MessageReceiverForSource_h
#define FWCore_Framework_MessageReceiverForSource_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     MessageReceiverForSource
// 
/**\class MessageReceiverForSource MessageReceiverForSource.h FWCore/Framework/interface/MessageReceiverForSource.h

 Description: Handled communication between controller process and worker processes when using multicore

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Dec 30 10:09:44 CST 2010
// $Id: MessageReceiverForSource.h,v 1.3 2011/10/24 22:27:16 chrjones Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
   namespace multicore {
      class MessageReceiverForSource
      {
         
      public:
         ///Takes the fd of the read and write socket for communication with parent
         MessageReceiverForSource(int parentSocket, int parentPipe);
         
         // ---------- const member functions ---------------------
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         
         /**Waits for a message on the queue. Throws a cms::Exception if there is an error */
         void receive();

         ///After calling receive this holds the index to the first event to process 
         unsigned long startIndex() const {
            return m_startIndex;
         }
         /** After calling receive this holds the number of consecutive event indices to be processed.
          It will return 0 if there is no more work to be done and the process should exit.
          */
         unsigned long numberOfConsecutiveIndices() const {
            return m_numberOfConsecutiveIndices;
         }
         /**After calling receive this holds the difference between the previous startIndex + numberOfConsecutiveIndices
          and the new startIndex. This is useful since most sources use 'skip' to go to next starting point.*/
         unsigned long numberToSkip() const {
            return m_numberToSkip;
         }
         
      private:
         MessageReceiverForSource(const MessageReceiverForSource&); // stop default
         
         const MessageReceiverForSource& operator=(const MessageReceiverForSource&); // stop default
         
         // ---------- member data --------------------------------
         int m_parentSocket;
         int m_parentPipe;
         int m_maxFd;
         unsigned long m_startIndex;
         unsigned long m_numberOfConsecutiveIndices;
         unsigned long m_numberToSkip;
         
      };
   }
}


#endif
