#include <iostream>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "IOPool/Streamer/interface/StreamerFileIO.h"


using namespace std;

int main()
{ 
  typedef vector<uint8> Buffer;
  Buffer buf(1024);
  Buffer buf2(1024);

  // ----------- init

  char psetid[] = "1234567890123456";
  char test_value[] = "This is a test, This is a";
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

  char reltag[]="CMSSW_0_6_0_pre45";

  InitMsgBuilder init(&buf[0],buf.size(),12,
                      Version(2,(const uint8*)psetid),(const char*)reltag,
                      hlt_names,l1_names);

  init.setDescLength(sizeof(test_value));
  std::copy(&test_value[0],&test_value[0]+sizeof(test_value),
            init.dataAddress());

  //Do a dumpInit here if you need to see the event.    


  //Start the Streamer file
  cout<<"Trying to Write a Streamer file"<<endl; 
  string initfilename = "teststreamfile.dat";
  StreamerOutputFile stream_writer(initfilename);
  
  //Start Index file
  cout<< "Trying to Write Out The Index Binary File" << initfilename << endl;
  string indexfilename = "testindexfile.ind";
  StreamerOutputIndexFile index_writer(indexfilename);

  cout<< "Trying to Write Out The Init message into Streamer File: " << initfilename << endl;
  stream_writer.write(init);

  cout<< "Trying to Write Out The Init message into Index File: "<<
                                                   indexfilename<<endl;
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

  //std::vector<EventMsgBuilder> myPreciousEvents;

  //Lets Build 10 Events ad then Write them into Styreamer file.

  //cout<<"ONLY Writting ONE Event"<<endl;
  //for (uint32 eventId = 2000; eventId != 2001; ++eventId) {
  for (uint32 eventId = 2000; eventId != 2010; ++eventId) {
    EventMsgBuilder emb(&buf[0],buf.size(),45,eventId,2,
                      l1bit,hltbits,hltsize);            
    emb.setReserved(78);
    emb.setEventLength(sizeof(test_value));
    std::copy(&test_value[0],&test_value[0]+sizeof(test_value),
             emb.eventAddr());

    //myPreciousEvents.push_back(emb);
  
    //Also lets write this to our streamer file too.
    cout<<"Writting Event# : "<<eventId<<" To Streamer file"<<endl;
    uint64 offset = stream_writer.write(emb);
  
    //Dummy Event Offset
    //long long int offset = (long long int) eventId;
    cout<<"Writting Event Index :"<<eventId<<" with offset# : "<<offset<<" To Index file"<<endl;
    index_writer.write(emb, (long long)offset);
  }


  //Write the EOF
  //EOFRecordBuilder eof = 
  //stream_writer.writeEOF();

 
  return 0;
}


