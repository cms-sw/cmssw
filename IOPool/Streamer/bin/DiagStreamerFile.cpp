/** Sample code to Read Streammer and recover file without bad events
    This is meant to debug streamer data files.

   compares_bad:
       Compares the streamer header info for two events and return true
       if any header information that should be the same is different

  uncompressBuffer:
       Tries to uncompress the event data blob if it was compressed
       and return true if successful (or was not compressed)

  readfile:
       Reads a streamer file, dumps the headers for the INIT message
       and the first event, and then looks to see if there are any
       events with streamer header problems or uncompress problems
       optionally writes a streamer file without bad events

  main():

      Code entry point, comment the function call that you don't want to make.

*/

#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/DumpTools.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "IOPool/Streamer/interface/StreamerOutputFile.h"

#include "zlib.h"

#include <iostream>
#include <memory>

bool compares_bad(EventMsgView const* eview1, EventMsgView const* eview2);
bool uncompressBuffer(unsigned char* inputBuffer,
                              unsigned int inputSize,
                              std::vector<unsigned char> &outputBuffer,
                              unsigned int expectedFullSize);
bool test_chksum(EventMsgView const* eview);
bool test_uncompress(EventMsgView const* eview, std::vector<unsigned char> &dest);
void readfile(std::string filename, std::string outfile);
void help();

//==========================================================================
int main(int argc, char* argv[]){

  if(argc < 2) {
    std::cout << "No command line argument supplied\n";
    help();
    return 1;
  }

  std::string streamfile(argv[1]);
  std::string outfile("/dev/null");
  if(argc == 3) {
    outfile = argv[2];
  }

  readfile(streamfile, outfile);
  std::cout << "\n\nDiagStreamerFile TEST DONE\n" << std::endl;

  return 0;
}

//==========================================================================
void help() {
      std::cout << "Usage: DiagStreamerFile streamer_file_name"
                << " [output_file_name]" << std::endl;
}
//==========================================================================
void readfile(std::string filename, std::string outfile) {

  uint32 num_events(0);
  uint32 num_badevents(0);
  uint32 num_baduncompress(0);
  uint32 num_badchksum(0);
  uint32 num_goodevents(0);
  uint32 num_duplevents(0);
  std::vector<unsigned char> compress_buffer(7000000);
  std::map<uint64, uint32> seenEventMap;
  bool output(false);
  if(outfile != "/dev/null") {
    output = true;
  }
  StreamerOutputFile stream_output(outfile);
  try{
    // ----------- init
    edm::StreamerInputFile stream_reader(filename);
    //if(output) StreamerOutputFile stream_output(outfile);

    std::cout << "Trying to Read The Init message from Streamer File: " << std::endl
         << filename << std::endl;
    InitMsgView const* init = stream_reader.startMessage();
    std::cout << "\n\n-------------INIT Message---------------------" << std::endl;
    std::cout << "Dump the Init Message from Streamer:-" << std::endl;
    dumpInitView(init);
    if(output) {
      stream_output.write(*init);
    }

    // ------- event
    std::cout << "\n\n-------------EVENT Messages-------------------" << std::endl;

    bool first_event(true);
    std::auto_ptr<EventMsgView> firstEvtView(0);
    std::vector<unsigned char> savebuf(0);
    EventMsgView const* eview(0);
    seenEventMap.clear();

    while(stream_reader.next()) {
      eview = stream_reader.currentRecord();
      ++num_events;
      bool good_event(true);
      if(seenEventMap.find(eview->event()) == seenEventMap.end()) {
         seenEventMap.insert(std::make_pair(eview->event(), 1));
      } else {
         ++seenEventMap[eview->event()];
         ++num_duplevents;
         std::cout << "??????? duplicate event Id for count " << num_events
                    << " event number " << eview->event()
                    << " seen " << seenEventMap[eview->event()] << " times" << std::endl;
      }
      if(first_event) {
        std::cout << "----------dumping first EVENT-----------" << std::endl;
        dumpEventView(eview);
        first_event = false;
        unsigned char* src = (unsigned char*)eview->startAddress();
        unsigned int srcSize = eview->size();
        savebuf.resize(srcSize);
        std::copy(src, src+srcSize, &(savebuf)[0]);
        firstEvtView.reset(new EventMsgView(&(savebuf)[0]));
        //firstEvtView, reset(new EventMsgView((void*)eview->startAddress()));
        if(!test_chksum(eview)) {
          std::cout << "checksum error for count " << num_events
                    << " event number " << eview->event()
                    << " from host name " << eview->hostName() << std::endl;
          ++num_badchksum;
          std::cout << "----------dumping bad checksum EVENT-----------" << std::endl;
          dumpEventView(eview);
          good_event = false;
        }
        if(!test_uncompress(eview, compress_buffer)) {
          std::cout << "uncompress error for count " << num_events
                    << " event number " << firstEvtView->event() << std::endl;
          ++num_baduncompress;
          std::cout << "----------dumping bad uncompress EVENT-----------" << std::endl;
          dumpEventView(firstEvtView.get());
          good_event = false;
        }
      } else {
        if(compares_bad(firstEvtView.get(), eview)) {
          std::cout << "Bad event at count " << num_events << " dumping event " << std::endl
                    << "----------dumping bad EVENT-----------" << std::endl;
          dumpEventView(eview);
          ++num_badevents;
          good_event = false;
        }
        if(!test_chksum(eview)) {
          std::cout << "checksum error for count " << num_events
                    << " event number " << eview->event()
                    << " from host name " << eview->hostName() << std::endl;
          ++num_badchksum;
          std::cout << "----------dumping bad checksum EVENT-----------" << std::endl;
          dumpEventView(eview);
          good_event = false;
        }
        if(!test_uncompress(eview, compress_buffer)) {
          std::cout << "uncompress error for count " << num_events
                    << " event number " << eview->event() << std::endl;
          ++num_baduncompress;
          std::cout << "----------dumping bad uncompress EVENT-----------" << std::endl;
          dumpEventView(eview);
          good_event = false;
        }
      }
      if(good_event) {
        if(output) {
          ++num_goodevents;
          stream_output.write(*eview);
        }
        //dumpEventView(eview);
      }
      if((num_events % 50) == 0) {
        std::cout << "Read " << num_events << " events, and "
                  << num_badevents << " events with bad headers, and "
                  << num_badchksum << " events with bad check sum, and "
                  << num_baduncompress << " events with bad uncompress" << std::endl;
        if(output) std::cout << "Wrote " << num_goodevents << " good events " << std::endl;
      }
    }

    std::cout << std::endl << "------------END--------------" << std::endl
              << "read " << num_events << " events" << std::endl
              << "and " << num_badevents << " events with bad headers" << std::endl
              << "and " << num_badchksum << " events with bad check sum"  << std::endl
              << "and " << num_baduncompress << " events with bad uncompress" << std::endl
              << "and " << num_duplevents << " duplicated event Id" << std::endl;

    if(output) {
      std::cout << "Wrote " << num_goodevents << " good events " << std::endl;
    }

  }catch(cms::Exception& e){
     std::cerr << "Exception caught:  "
               << e.what() << std::endl
               << "After reading " << num_events << " events, and "
               << num_badevents << " events with bad headers" << std::endl
               << "and " << num_badchksum << " events with bad check sum"  << std::endl
               << "and " << num_baduncompress << " events with bad uncompress"  << std::endl
               << "and " << num_duplevents << " duplicated event Id" << std::endl;
  }
}

//==========================================================================
bool compares_bad(EventMsgView const* eview1, EventMsgView const* eview2) {
  bool is_bad(false);
  if(eview1->code() != eview2->code()) {
    std::cout << "non-matching EVENT message code " << std::endl;
    is_bad = true;
  }
  if(eview1->protocolVersion() != eview2->protocolVersion()) {
    std::cout << "non-matching EVENT message protocol version" << std::endl;
    is_bad = true;
  }
  if(eview1->run() != eview2->run()) {
    std::cout << "non-matching run number " << std::endl;
    is_bad = true;
  }
  if(eview1->lumi() != eview2->lumi()) {
    std::cout << "non-matching lumi number" << std::endl;
    is_bad = true;
  }
  if(eview1->outModId() != eview2->outModId()) {
    std::cout << "non-matching output module id" << std::endl;
    is_bad = true;
  }
  if(eview1->hltCount() != eview2->hltCount()) {
    std::cout << "non-matching HLT count" << std::endl;
    is_bad = true;
  }
  if(eview1->l1Count() != eview2->l1Count()) {
    std::cout << "non-matching L1 count" << std::endl;
    is_bad = true;
  }
  return is_bad;
}

//==========================================================================
bool test_chksum(EventMsgView const* eview) {
  uint32_t adler32_chksum = cms::Adler32((char const*)eview->eventData(), eview->eventLength());
  //std::cout << "Adler32 checksum of event = " << adler32_chksum << std::endl;
  //std::cout << "Adler32 checksum from header = " << eview->adler32_chksum() << std::endl;
  //std::cout << "event from host name = " << eview->hostName() << std::endl;
  if((uint32)adler32_chksum != eview->adler32_chksum()) {
    std::cout << "Bad chekcsum: Adler32 checksum of event data  = " << adler32_chksum
              << " from header = " << eview->adler32_chksum()
              << " host name = " << eview->hostName() << std::endl;
    return false;
  }
  return true;
}

//==========================================================================
bool test_uncompress(EventMsgView const* eview, std::vector<unsigned char> &dest) {
  unsigned long origsize = eview->origDataSize();
  bool success = false;
  if(origsize != 0 && origsize != 78)
  {
    // compressed
    success = uncompressBuffer(const_cast<unsigned char*>((unsigned char const*)eview->eventData()),
                                   eview->eventLength(), dest, origsize);
  } else {
    // uncompressed anyway
    success = true;
  }
  return success;
}

//==========================================================================
bool uncompressBuffer(unsigned char *inputBuffer,
                              unsigned int inputSize,
                              std::vector<unsigned char> &outputBuffer,
                              unsigned int expectedFullSize)
  {
    unsigned long origSize = expectedFullSize;
    unsigned long uncompressedSize = expectedFullSize*1.1;
    outputBuffer.resize(uncompressedSize);
    int ret = uncompress(&outputBuffer[0], &uncompressedSize,
                         inputBuffer, inputSize);
    if(ret == Z_OK) {
        // check the length against original uncompressed length
        if(origSize != uncompressedSize) {
            std::cout << "Problem with uncompress, original size = "
                 << origSize << " uncompress size = " << uncompressedSize << std::endl;
            return false;
        }
    } else {
        std::cout << "Problem with uncompress, return value = "
             << ret << std::endl;
        return false;
    }
    return true;
}

