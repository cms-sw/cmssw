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

#include <memory>
#include <iostream>
#include "zlib.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/DumpTools.h"

#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "IOPool/Streamer/interface/StreamerOutputFile.h"

#include "FWCore/Utilities/interface/Exception.h"

bool compares_bad(const EventMsgView* eview1, const EventMsgView* eview2);
bool uncompressBuffer(unsigned char *inputBuffer,
                              unsigned int inputSize,
                              std::vector<unsigned char> &outputBuffer,
                              unsigned int expectedFullSize);
bool test_uncompress(const EventMsgView* eview, std::vector<unsigned char> &dest);
void readfile(std::string filename, std::map<uint32, uint32> &dupEventMap);
void studyDupInFile(std::string filename, std::map<uint32, uint32> &dupEventMap);
bool sameEvtHeader(const EventMsgView* eview1, const EventMsgView* eview2);
unsigned int compEvtDataBlob(const EventMsgView* eview1, const EventMsgView* eview2);
void help();

//==========================================================================
int main(int argc, char* argv[]){

  if (argc < 2) {
    std::cout << "No command line argument supplied\n";
    help();
    return 0;
  }

  std::string streamfile(argv[1]);
  std::string outdump("none");
  if(argc == 3) {
    outdump = argv[2];
  }

  std::map<uint32, uint32> dupEventMap;
  dupEventMap.clear();
  readfile(streamfile, dupEventMap);
  if(!dupEventMap.empty()) studyDupInFile(streamfile, dupEventMap);
  std::cout <<"\n\nAnalDup TEST DONE\n"<< std::endl;

  return 0;
}

//==========================================================================
void help() {
      std::cout << "Usage: AnalDup streamer_file_name" << std::endl;
}

//==========================================================================
void readfile(std::string filename, std::map<uint32, uint32> &dupEventMap) {

  int num_events(0);
  int num_badevents(0);
  int num_baduncompress(0);
  int num_duplevents(0);
  std::vector<unsigned char> compress_buffer(7000000);
  try{
    // ----------- init
    edm::StreamerInputFile stream_reader (filename);

    std::cout << "Trying to Read The Init message from Streamer File: " << std::endl
         << filename << std::endl;
    const InitMsgView* init = stream_reader.startMessage();
    std::cout<<"\n\n-------------INIT Message---------------------"<< std::endl;
    std::cout<<"Dump the Init Message from Streamer:-"<< std::endl;
    dumpInitView(init);

    // ------- event
    std::cout<<"\n\n-------------EVENT Messages-------------------"<< std::endl;

    bool first_event(true);
    std::auto_ptr<EventMsgView> firstEvtView(0);
    std::vector<unsigned char> savebuf(0);
    const EventMsgView* eview(0);
    std::map<uint32, uint32> seenEventMap;
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
         if(dupEventMap.find(eview->event()) == dupEventMap.end()) {
           dupEventMap.insert(std::make_pair(eview->event(), 2));
         } else {
           ++dupEventMap[eview->event()];
         }
         std::cout << "??????? duplicate event Id for count " << num_events
                    << " event number " << eview->event()
                    << " seen " << seenEventMap[eview->event()] << " times" << std::endl;
          std::cout<<"----------dumping duplicated EVENT-----------"<< std::endl;
          dumpEventView(eview);
      }
      if(first_event) {
        std::cout<<"----------dumping first EVENT-----------"<< std::endl;
        dumpEventView(eview);
        first_event = false;
        unsigned char* src = (unsigned char*)eview->startAddress();
        unsigned int srcSize = eview->size();
        savebuf.resize(srcSize);
        std::copy(src, src+srcSize, &(savebuf)[0]);
        firstEvtView.reset(new EventMsgView(&(savebuf)[0]));
        //firstEvtView.reset(new EventMsgView((void*)eview->startAddress()));
        //std::cout<<"----------dumping copied first EVENT-----------"<< std::endl;
        //dumpEventView(firstEvtView.get());
        if(!test_uncompress(firstEvtView.get(), compress_buffer)) {
          std::cout << "uncompress error for count " << num_events 
                    << " event number " << firstEvtView->event() << std::endl;
          ++num_baduncompress;
          std::cout<<"----------dumping bad uncompress EVENT-----------"<< std::endl;
          dumpEventView(firstEvtView.get());
          good_event=false;
        }
      } else {
        if(compares_bad(firstEvtView.get(), eview)) {
          std::cout << "Bad event at count " << num_events << " dumping event " << std::endl
                    << "----------dumping bad EVENT-----------"<< std::endl;
          dumpEventView(eview);
          ++num_badevents;
          good_event=false;
        }
        if(!test_uncompress(eview, compress_buffer)) {
          std::cout << "uncompress error for count " << num_events 
                    << " event number " << eview->event() << std::endl;
          ++num_baduncompress;
          std::cout<<"----------dumping bad uncompress EVENT-----------"<< std::endl;
          dumpEventView(eview);
          good_event=false;
        }
      }
      if((num_events % 50) == 0) {
        std::cout << "Read " << num_events << " events, and "
                  << num_badevents << " events with bad headers, and "
                  << num_baduncompress << " events with bad uncompress" << std::endl;
      }
    }
    std::cout << std::endl << "------------END--------------" << std::endl
              << "read " << num_events << " events" << std::endl
              << "and " << num_badevents << " events with bad headers" << std::endl
              << "and " << num_baduncompress << " events with bad uncompress" << std::endl
              << "and " << num_duplevents << " duplicated event Id" << std::endl;

  }catch (cms::Exception& e){
     std::cerr << "Exception caught:  "
               << e.what() << std::endl
               << "After reading " << num_events << " events, and "
               << num_badevents << " events with bad headers" << std::endl
               << "and " << num_baduncompress << " events with bad uncompress"  << std::endl
               << "and " << num_duplevents << " duplicated event Id" << std::endl;
  }
}

//==========================================================================
bool compares_bad(const EventMsgView* eview1, const EventMsgView* eview2) {
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
bool test_uncompress(const EventMsgView* eview, std::vector<unsigned char> &dest) {
  unsigned long origsize = eview->origDataSize();
  bool success = false;
  if(origsize != 0)
  {
    // compressed
    success = uncompressBuffer((unsigned char*)eview->eventData(),
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
//==========================================================================
void studyDupInFile(std::string filename, std::map<uint32, uint32> &dupEventMap) {

  int num_events(0);
  std::map<uint32, EventMsgView*> dupEventDataMap;
  std::vector<unsigned char> compress_buffer(7000000);
  try{
    // ----------- init
    edm::StreamerInputFile stream_reader (filename);

    std::cout << std::endl << "Trying to Study duplicate events in Streamer File: " << std::endl
         << filename << std::endl;
    //const InitMsgView* init = stream_reader.startMessage();
    //std::cout<<"\n\n-------------INIT Message---------------------"<< std::endl;
    //std::cout<<"Dump the Init Message from Streamer:-"<< std::endl;
    //dumpInitView(init);

    // ------- event
    std::cout<<"\n\n-------------EVENT Messages-------------------"<< std::endl;

    const EventMsgView* eview(0);
  
    std::cout << "Number duplicated events = " << dupEventMap.size() << std::endl;
    while(stream_reader.next()) {
      eview = stream_reader.currentRecord();
      ++num_events;
      if(dupEventMap.find(eview->event()) != dupEventMap.end()) {
        // this event was duplicated
        if(dupEventDataMap.find(eview->event()) == dupEventDataMap.end()) {
          // first instance of this duplicated event - save it
          std::cout << "Saving duplicate event at count " << num_events
                    << " event number " << eview->event()
                    << " seen " << dupEventMap[eview->event()] << " times" << std::endl;
          //std::cout<<"----------dumping EVENT-----------"<< std::endl;
          //dumpEventView(eview);
          std::vector<unsigned char> savebuf(0);
          unsigned char* src = (unsigned char*)eview->startAddress();
          unsigned int srcSize = eview->size();
          savebuf.resize(srcSize);
          std::copy(src, src+srcSize, &(savebuf)[0]);
          EventMsgView saveEvtView(&(savebuf)[0]);
          //std::cout<<"----------dumping saved EVENT-----------"<< std::endl;
          //dumpEventView(&saveEvtView);
          dupEventDataMap.insert(std::make_pair(eview->event(), &saveEvtView));
        } else {
          // next instance of duplicated event test it
          std::cout << "Testing duplicate event at count " << num_events
                    << " event number " << eview->event()
                    << " seen " << dupEventMap[eview->event()] << " times" << std::endl;
          //std::cout<<"----------dumping EVENT-----------"<< std::endl;
          //dumpEventView(eview);
          if(sameEvtHeader(eview, dupEventDataMap[eview->event()])) {
            // header is the same look at data blob
            std::cout << "Duplicate event has same header checking data blob" << std::endl;
            unsigned int bytesDiff = compEvtDataBlob(eview, dupEventDataMap[eview->event()]);
            if(bytesDiff == 0) {
              // data blob is the same
              std::cout << "Duplicate event has same data blob" << std::endl;
            } else {
              // data blob is different
              std::cout << "Duplicate event data blob has " << bytesDiff << " bytes difference" << std::endl;
            }
          } else {
            std::cout << "Duplicate event has different header" << std::endl;
          }
        }
       }
      if((num_events % 1000) == 0) {
        std::cout << "Read " << num_events << " events" << std::endl;
      }
    }
    std::cout << std::endl << "------------END--------------" << std::endl
              << "read " << num_events << " events" << std::endl;
  }catch (cms::Exception& e){
     std::cerr << "Exception caught:  "
               << e.what() << std::endl
               << "After reading " << num_events << " events" << std::endl;
  }
}

//==========================================================================
bool sameEvtHeader(const EventMsgView* eview1, const EventMsgView* eview2) {
  bool is_same(true);
  if(eview1->code() != eview2->code()) {
    std::cout << "non-matching EVENT message code " << std::endl;
    is_same = false;
  }
  if(eview1->size() != eview2->size()) {
    std::cout << "non-matching EVENT message size " << std::endl;
    is_same = false;
  }
  if(eview1->eventLength() != eview2->eventLength()) {
    std::cout << "non-matching EVENT message data length" << std::endl;
    is_same = false;
  }
  if(eview1->headerSize() != eview2->headerSize()) {
    std::cout << "non-matching EVENT message header size " << std::endl;
    is_same = false;
  }
  if(eview1->protocolVersion() != eview2->protocolVersion()) {
    std::cout << "non-matching EVENT message protocol version" << std::endl;
    is_same = false;
  }
  if(eview1->run() != eview2->run()) {
    std::cout << "non-matching run number" << std::endl;
    is_same = false;
  }
  if(eview1->event() != eview2->event()) {
    std::cout << "non-matching event numbercode " << std::endl;
    is_same = false;
  }
  if(eview1->lumi() != eview2->lumi()) {
    std::cout << "non-matching lumi number" << std::endl;
    is_same = false;
  }
  if(eview1->origDataSize() != eview2->origDataSize()) {
    std::cout << "non-matching original data size" << std::endl;
    is_same = false;
  }
  if(eview1->outModId() != eview2->outModId()) {
    std::cout << "non-matching EVENT message code " << std::endl;
    is_same = false;
  }
  if(eview1->hltCount() != eview2->hltCount()) {
    std::cout << "non-matching EVENT message code " << std::endl;
    is_same = false;
  } else {
    if (eview1->hltCount() > 0) {
      std::vector<unsigned char> hlt_out1;
      hlt_out1.resize(1 + (eview1->hltCount()-1)/4);
      eview1->hltTriggerBits(&hlt_out1[0]);
      std::vector<unsigned char> hlt_out2;
      hlt_out2.resize(1 + (eview2->hltCount()-1)/4);
      eview2->hltTriggerBits(&hlt_out2[0]);
      //bool result = hlt_out1 == hlt_out2;
      bool result = std::equal( hlt_out1.begin(), hlt_out1.end(), hlt_out2.begin() );
      if(!result) {
        std::cout << "non-matching HLT trigger bits" << std::endl;
        is_same = false;
      }
    }
  }
  if(eview1->l1Count() != eview2->l1Count()) {
    std::cout << "non-matching EVENT message code " << std::endl;
    is_same = false;
  } else {
    std::vector<bool> l1_out1;
    eview1->l1TriggerBits(l1_out1);
    std::vector<bool> l1_out2;
    eview2->l1TriggerBits(l1_out2);
      bool result = l1_out1 == l1_out2;
      //bool result = std::equal( hlt_out1.begin(), hlt_out1.end(), hlt_out2.begin() );
      if(!result) {
        std::cout << "non-matching L1 trigger bits" << std::endl;
        is_same = false;
      }
  }
  return is_same;
}

//==========================================================================
unsigned int compEvtDataBlob(const EventMsgView* eview1, const EventMsgView* eview2) {
  unsigned int bytes_diff(0);
  if(eview1->eventLength() != eview2->eventLength()) {
    std::cout << "non-matching EVENT message data length cannot continue" << std::endl;
    if(eview1->eventLength() > eview2->eventLength())
      return (eview1->eventLength() - eview2->eventLength());
    else
      return (eview2->eventLength() - eview1->eventLength());
  }
  if (eview1->eventLength() > 0) {
    unsigned int size = eview1->eventLength();
    uint8 *datablob1 = eview1->startAddress();
    uint8 *datablob2 = eview2->startAddress();
    for (unsigned i=0; i< size; i++) {
       if(*(datablob1+i) != *(datablob2+i)) ++bytes_diff;
    }
  }
  return bytes_diff;
}
