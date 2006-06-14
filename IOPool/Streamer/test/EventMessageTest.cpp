/****
   This code basically performs following steps

   // make a buffer area to "serial" the information
   vector<unsigned char> buf(1024);

   // write the header in to the buffer during construction
   InitMsgBuilder init_build(&buf[0],buf.size(), ... all the data for the header);
   // write the registry directly after the header
   unsigned int bytes_written = serialize_registry_here(registry_thing, init_build.dataAddress());
   init_build.setDescLength(bytes_written);
     // write an event header into the buffer during construction
   EventMsgBuilder evt_build(&buf[0],buf.size(), ... all the header data);
   // write the event data after the header
   unsigned int bytes_written = serial_event_here(event,evt_build.eventData());
   evt_build.setEventLength(bytes_written);

   // look at the init message
   InitMsgView init_view(&buf[0],buf.size());
   // use member function to pull out the data

   // look at the event message
   EventMsgView evt_view(&buf[0],buf.size());
   // use member functions to pull out the data

*****/



#include <iostream>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

using namespace std;

void dumpInit(uint8* buf, uint32 bufsize)
{
  InitMsgView view(buf,bufsize);

  cout
    << "code = " << view.code() << ", "
    << "size = " << view.size() << "\n"
    << "run = " << view.run() << ", "
    << "proto = " << view.protocolVersion() << "\n"
    << "release = " << view.releaseTag() << "\n";

  uint8 vpset[17];
  view.pset(vpset);
  vpset[16]='\0';
  Strings vhltnames,vl1names;
  view.hltTriggerNames(vhltnames);
  view.l1TriggerNames(vl1names);

  cout << "pset = " << vpset << "\n";
  cout << "\nHLT names = \n";
  copy(vhltnames.begin(),vhltnames.end(),ostream_iterator<string>(cout,"\n"));
  cout << "\nL1 names = \n";
  copy(vl1names.begin(),vl1names.end(),ostream_iterator<string>(cout,"\n"));
  cout << "\n";

  cout << "desc len = " << view.descLength() << "\n";
  const uint8* pos = view.descData();
  copy(pos,pos+view.descLength(),ostream_iterator<uint8>(cout,""));
  cout << "\n";
}

void dumpEvent(uint8* buf,uint32 bufsize,uint32 hltsize,uint32 l1size)
{
  EventMsgView eview(buf,bufsize,hltsize,l1size);

  cout << "----------------------\n";
  cout << "code=" << eview.code() << "\n"
       << "size=" << eview.size() << "\n"
       << "run=" << eview.run() << "\n"
       << "event=" << eview.event() << "\n"
       << "lumi=" << eview.lumi() << "\n"
       << "reserved=" << eview.reserved() << "\n"
       << "event length=" << eview.eventLength() << "\n";

  std::vector<bool> l1_out;
  uint8 hlt_out[10];
  eview.l1TriggerBits(l1_out);
  eview.hltTriggerBits(hlt_out);

  cout << "\nl1 size= " << l1_out.size() << " l1 bits=\n";
  copy(l1_out.begin(),l1_out.end(),ostream_iterator<bool>(cout," "));

  cout << "\nhlt bits=\n(";
  copy(&hlt_out[0],&hlt_out[0]+hltsize/4,ostream_iterator<char>(cout,""));
  cout << ")\n";

  const uint8* edata = eview.eventData();
  cout << "\nevent data=\n(";
  copy(&edata[0],&edata[0]+eview.eventLength(),
       ostream_iterator<char>(cout,""));
  cout << ")\n";

}


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

  cout<<"\n\nAnzar:::Dumping Init Msg::\n"<<endl;
  dumpInit(&buf[0],buf.size());

  InitMsgView view(&buf[0],buf.size());
  uint8 psetid2[16];
  Strings hlt2;
  Strings l12;
  view.pset(psetid2);
  view.hltTriggerNames(hlt2);
  view.l1TriggerNames(l12);

  InitMsgBuilder init2(&buf2[0],buf2.size(),
                       view.run(),
                       Version(view.protocolVersion(),
                               (const uint8*)psetid2),
                       view.releaseTag().c_str(),
                       hlt2,l12);

  init2.setDescLength(view.descLength());
  std::copy(view.descData(),view.descData()+view.size(),
            init2.dataAddress());

  cout<<"\n\nAnzar:::Dumping Init Msg After Copying::\n"<<endl;
  dumpInit(&buf2[0],buf2.size());

  if(equal(&buf[0],&buf[0]+view.size(),buf2.begin())==false)
    {
      cerr << "Init buffers not the same!\n";
      dumpInit(&buf[0],buf.size());
      dumpInit(&buf2[0],buf2.size());
      abort();
    }

  // ------- event

  std::vector<bool> l1bit(16);
  uint8 hltbits[] = "4567";
  const int hltsize = (sizeof(hltbits)-1)*4;

  l1bit[0]=true;  l1bit[4]=true;  l1bit[8]=false;  l1bit[12]=true;
  l1bit[1]=true;  l1bit[5]=false;  l1bit[9]=false;  l1bit[13]=false;
  l1bit[2]=false;  l1bit[6]=true;  l1bit[10]=true;  l1bit[14]=false;
  l1bit[3]=false;  l1bit[7]=false;  l1bit[11]=true;  l1bit[15]=true;

  EventMsgBuilder emb(&buf[0],buf.size(),45,2020,2,
                      l1bit,hltbits,hltsize);

  emb.setReserved(78);
  emb.setEventLength(sizeof(test_value));
  std::copy(&test_value[0],&test_value[0]+sizeof(test_value),
            emb.eventAddr());

  cout<<"Anzar: Dumping Event:\n"<<endl;
  cout << "l1bit size input=" << l1bit.size() << "\n";
  dumpEvent(&buf[0],buf.size(),hltsize,l1bit.size());

  // ------ check for sameness

  EventMsgView eview(&buf[0],buf.size(),hltsize,l1bit.size());

  std::vector<bool> l1_out;
  uint8 hlt_out[10];
  eview.l1TriggerBits(l1_out);
  eview.hltTriggerBits(hlt_out);

  EventMsgBuilder emb2(&buf2[0],buf.size(),
                       eview.run(),
                       eview.event(),
                       eview.lumi(),
                       l1_out,
                       hlt_out,
                       hltsize);

  emb2.setReserved(eview.reserved());
  emb2.setEventLength(eview.eventLength());
  std::copy(eview.eventData(),eview.eventData()+eview.eventLength(),
            emb2.eventAddr());

  cout<<"Anzar: Dumping Recreated Event:\n"<<endl;
  dumpEvent(&buf2[0],buf2.size(),hltsize,l1_out.size());

  if(equal(&buf[0],&buf[0]+emb.size(),buf2.begin())==false)
    {
      cerr << "event messages not the same\n";
      cerr << "size 1st=" << emb.size() << "\n"
           << "size 2nd=" << emb2.size() << "\n";

      dumpEvent(&buf[0],buf.size(),hltsize,l1bit.size());
      dumpEvent(&buf2[0],buf2.size(),hltsize,l1_out.size());
      abort();
    }
  return 0;
}


