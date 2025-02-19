/*

Example code to write DQM Event Message to file

*/

#include <iostream>
#include <stdint.h>

#include "IOPool/Streamer/interface/MsgTools.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/StreamDQMOutputFile.h"

#include "DataFormats/Provenance/interface/Timestamp.h"

using namespace edm;

//namespace edm
//{

int main()
{
  typedef std::vector<uint8> Buffer;
  Buffer buf(1024);
  //Buffer buf2(1024);

  uint32 run = 123;
  uint32 event = 456;
  uint32 lumiSection = 789;
  uint32 updateNumber = 111;
  char topFolderName[]="TopFolder";
  uint32_t adler32_chksum = 42;
  std::string host_name = "mytestnode.cms";

  DQMEvent::TObjectTable toTable;

  StreamDQMOutputFile dqmFile("dqm_file.dqm");

  try {
      for (int i=0; i != 10; ++i) {
        Timestamp fakeTime(i);
        // create the message
        DQMEventMsgBuilder dqmMsgBuilder(&buf[0], buf.size(),
                                     run, event++,
                                     //fakeTime.value(),
                                     fakeTime,
                                     lumiSection, updateNumber,
                                     (uint32)adler32_chksum,
                                     host_name.c_str(),
                                     edm::getReleaseVersion(), topFolderName,
                                     toTable);


        //Serialize the msg


        //Lets write out this msg in a file

        dqmFile.write(dqmMsgBuilder);
     }
  }
  catch(cms::Exception const& e) {
    std::cout << e.explainSelf() << std::endl;
    return 1;
  }
  /*** DQMEventMsgView dqmMsgView(&buf[0]);
  std::cout
    << "code = " << dqmMsgView.code()<< ", "
    << "\nsize = " << dqmMsgView.size()<< ", "
    << "\nrun = " << dqmMsgView.runNumber() << ", "
    << "\ntimeStamp = " << dqmMsgView.timeStamp().value() << " "
    << "\neventLength = " << dqmMsgView.eventLength() << ", "
    << "\nevent = " << dqmMsgView.eventNumberAtUpdate() << "\n"
    << "topFolderName = " << dqmMsgView.topFolderName() << "\n"
    << "release = " << dqmMsgView.releaseTag() << "\n";
   ***/
  return 0;
}

//}
