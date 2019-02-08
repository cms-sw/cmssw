#ifndef FWCore_Framework_MessageForSource_h
#define FWCore_Framework_MessageForSource_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     MessageForSource
//
/**\class MessageForSource MessageForSource.h FWCore/Framework/interface/MessageForSource.h

 Description: Information passed from controller to source when doing multicore processing

 Usage:
    This class is an internal detail of how the parent process communicates to child processes.
 It is used with the posix sockets to send what events child processes should handle. The
 events are designated as a 'block' where we give the index to the first event in the block and
 then the number of consecutive events to process in the block.

 NOTE: If the number of consecutive events (i.e. nIndices) is 0, this means there is no more
 work to do and the child process should end.
 
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Dec 30 10:08:24 CST 2010
//

// system include files
#include <cstddef>
// user include files

// forward declarations

namespace edm {
  namespace multicore {
    class MessageForSource {
    public:
      MessageForSource() : startIndex(0), nIndices(0) {}

      //virtual ~MessageForSource();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      static size_t sizeForBuffer() { return sizeof(MessageForSource); }

      // ---------- member functions ---------------------------

    public:
      //MessageForSource(const MessageForSource&); // allow default

      //const MessageForSource& operator=(const MessageForSource&); // allow default

      // ---------- member data --------------------------------
      unsigned long startIndex;  //which event index to start processing for this 'block'
      unsigned long nIndices;    //number of consecutive indicies in the block
    };
  }  // namespace multicore
}  // namespace edm

#endif
