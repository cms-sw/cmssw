/* 

Example code to write DQM Event Message to file

*/

#include <iostream>

#include "IOPool/Streamer/interface/MsgTools.h"
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

  DQMEvent::TObjectTable toTable;
  
  StreamDQMOutputFile dqmFile("dqm_file.dqm");

  for (int i=0; i != 10; ++i) {
	Timestamp fakeTime(i);
        // create the message
  	DQMEventMsgBuilder dqmMsgBuilder(&buf[0], buf.size(),
                                     run, event++, 
				     //fakeTime.value(),
				     fakeTime,
                                     lumiSection, updateNumber,
                                     edm::getReleaseVersion(), topFolderName,
                                     toTable);


  	//Serialize the msg
  

  	//Lets write out this msg in a file

  	dqmFile.write(dqmMsgBuilder);
   } 
  /***	DQMEventMsgView dqmMsgView(&buf[0]);
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
