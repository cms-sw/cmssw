/** Example code shows you to write Streamer and Index files.
    All values are dummy here, The Init message contains different 
    values from what Event Header contains, this is only 
    for the demonstration, obviously.

    Change total number of written events using,

     #define NO_OF_EVENTS 10

    Crank it up to 10000 if you like to scrol your screen ;-).

Disclaimer: Most of the code here is randomly written during
               testing various parts, its not a supported testing code.
               Changes can and will be made, when and if required.
*/

#include <iostream>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"

#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "IOPool/Streamer/interface/StreamerOutputIndexFile.h"

#include "zlib.h"

#define NO_OF_EVENTS 10

int main()
{ 
  typedef std::vector<uint8> Buffer;
  Buffer buf(1024);
  Buffer buf2(1024);

  // ----------- init

  char psetid[] = "1234567890123456";
  char test_value[] = "This is a test, This is a";
  char test_value_event[] = "This is a test Event, This is a";
  Strings hlt_names;
  Strings l1_names;

  hlt_names.push_back("a");  hlt_names.push_back("b");
  hlt_names.push_back("c");  hlt_names.push_back("d");
  hlt_names.push_back("e");  hlt_names.push_back("f");
  hlt_names.push_back("g");  hlt_names.push_back("h");
  hlt_names.push_back("i");
    
  l1_names.push_back("t10");  l1_names.push_back("t11");
  l1_names.push_back("t12");  l1_names.push_back("t13");
  l1_names.push_back("t14");  l1_names.push_back("t15");
  l1_names.push_back("t16");  l1_names.push_back("t17");
  l1_names.push_back("t18");  l1_names.push_back("t19");
  l1_names.push_back("t20");

  char reltag[]="CMSSW_0_8_0_pre7";
  std::string processName = "HLT";
  std::string outputModuleLabel = "HLTOutput";

  uLong crc = crc32(0L, Z_NULL, 0);
  Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
  crc = crc32(crc, crcbuf, outputModuleLabel.length());

  InitMsgBuilder init(&buf[0],buf.size(),12,
                      Version(7,(const uint8*)psetid),
                      (const char*)reltag, processName.c_str(),
                      outputModuleLabel.c_str(), crc,
                      hlt_names,hlt_names,l1_names);

  init.setDataLength(sizeof(test_value));
  std::copy(&test_value[0],&test_value[0]+sizeof(test_value),
            init.dataAddress());

  //Do a dumpInit here if you need to see the event.    

  //Start the Streamer file
  std::cout <<"Trying to Write a Streamer file"<< std::endl; 
  std::string initfilename = "teststreamfile.dat";
  StreamerOutputFile stream_writer(initfilename);
  
  //Start Index file
  std::cout << "Trying to Write Out The Index Binary File" << initfilename << std::endl;
  std::string indexfilename = "testindexfile.ind";
  StreamerOutputIndexFile index_writer(indexfilename);

  std::cout << "Trying to Write Out The Init message into Streamer File: " 
      << initfilename << std::endl;
  stream_writer.write(init);

  std::cout << "Trying to Write Out The Init message into Index File: "<<
                                                   indexfilename<< std::endl;
  uint32 magic = 22;
  uint64 reserved = 666;
  index_writer.writeIndexFileHeader(magic, reserved);
  index_writer.write(init);

  // ------- event

  std::vector<bool> l1bit(11);
  uint8 hltbits[] = "4567";
  //const int hltsize = (sizeof(hltbits)-1)*4;
  const int hltsize = 9;  /** I am interested in 9 bits only */

  l1bit[0]=true;  l1bit[4]=true;  l1bit[8]=false;  //l1bit[12]=true;
  l1bit[1]=true;  l1bit[5]=false;  l1bit[9]=false;  //l1bit[13]=false;
  l1bit[2]=false;  l1bit[6]=true;  l1bit[10]=true;  //l1bit[14]=false;
  l1bit[3]=false;  l1bit[7]=false;  l1bit[11]=true;  //l1bit[15]=true;
  //l1bit[16]=false;  l1bit[17]=false;  l1bit[18]=true;  l1bit[19]=true;

  //Lets Build 10 Events ad then Write them into Streamer/Index file.
  
  for (uint32 eventId = 2000; eventId != 2000+NO_OF_EVENTS; ++eventId) {
    EventMsgBuilder emb(&buf[0],buf.size(),45,eventId,2,0xdeadbeef,
                      l1bit,hltbits,hltsize);            
    emb.setOrigDataSize(78);
    emb.setEventLength(sizeof(test_value_event));
    std::copy(&test_value_event[0],&test_value_event[0]+sizeof(test_value_event),
             emb.eventAddr());

    //Lets write this to our streamer file .
    std::cout<<"Writing Event# : "<<eventId<<" To Streamer file"<< std::endl;
    uint64 offset = stream_writer.write(emb);
  
    //Lets write the Index too
    std::cout<<"Writing Event Index :" << eventId 
        <<" with offset# : " << offset << " To Index file" 
                             << std::endl;
    index_writer.write(emb, offset);
  }


  //Write the EOF Record Both at the end of Streamer file and Index file
  uint32 dummyStatusCode = 1234;
  std::vector<uint32> hltStats;

  hltStats.push_back(32);
  hltStats.push_back(33);
  hltStats.push_back(34);

  stream_writer.writeEOF(dummyStatusCode, hltStats);

  index_writer.writeEOF(dummyStatusCode, hltStats);

 
  return 0;
}


