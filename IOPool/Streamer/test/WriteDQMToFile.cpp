/* 

Example code to write DQM Event Message to file

*/

#include <iostream>

#include "IOPool/Streamer/interface/MsgTools.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/StreamDQMOutputFile.h"

#include "DataFormats/Provenance/interface/Timestamp.h"

using namespace std;
using namespace edm;

//namespace edm
//{

int main()
{ 
  typedef vector<uint8> Buffer;
  Buffer buf(1024);
  //Buffer buf2(1024);

  uint32 run = 123;
  uint32 event = 456;
  uint32 lumiSection = 789;
  uint32 updateNumber = 111;
  char topFolderName[]="TopFolder";

  //  Event.h, it has a time function, Timestamp time() const {return aux_.time();}
  //  FWCore/Framework/interface/Event.h

  Timestamp fakeTime(2);
  
  //long long ll;
  //cout << "Fake Timestamp:" << fakeTime.value() << endl;
  //cout << "SIZEOF Fake Timestamp:" << sizeof(fakeTime.value()) << endl;
  //cout << "SIZEOF 2_long:" << sizeof(ll) << endl;

  DQMEvent::TObjectTable toTable;
  // create the message
  DQMEventMsgBuilder dqmMsgBuilder(&buf[0], buf.size(),
                                     run, event, 
				     //fakeTime.value(),
				     fakeTime,
                                     lumiSection, updateNumber,
                                     edm::getReleaseVersion(), topFolderName,
                                     toTable);


  //Serialize the msg
  

  //Lets write out this msg in a file
  StreamDQMOutputFile dqmFile("dqm_file.dqm");


  dqmFile.write();


  DQMEventMsgView dqmMsgView(&buf[0]);
  cout
    << "code = " << dqmMsgView.code()<< ", "
    << "\nsize = " << dqmMsgView.size()<< ", "
    << "\nrun = " << dqmMsgView.runNumber() << ", "
    << "\ntimeStamp = " << dqmMsgView.timeStamp().value() << " "
    << "\neventLength = " << dqmMsgView.eventLength() << ", "
    << "\nevent = " << dqmMsgView.eventNumberAtUpdate() << "\n"
    << "topFolderName = " << dqmMsgView.topFolderName() << "\n"
    << "release = " << dqmMsgView.releaseTag() << "\n";

  //init.setDescLength(sizeof(test_value));
  //std::copy(&test_value[0],&test_value[0]+sizeof(test_value),
  //          init.dataAddress());

  //Do a dumpInit here if you need to see the event.    

  return 0;
}

//}
