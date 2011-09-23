/*sageTest and 
    Reads and comapres them.

   The Init/EventMessage(s) are ONLY dumped if they are different 
   (Shouldn't be in this original code).

Disclaimer: Most of the code here is randomly written during
               testing various parts, its not a supported testing code.
               Changes can and will be made, when and if required.

*/


#include <iostream>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/DumpTools.h"
#include "zlib.h"

int main()
{ 
  typedef std::vector<uint8> Buffer;
  Buffer buf(2024);
  Buffer buf2(2024);

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

  char reltag[]="CMSSW_0_8_0_pre7";
  std::string processName = "HLT";
  std::string outputModuleLabel = "HLTOutput";

  uLong crc = crc32(0L, Z_NULL, 0);
  Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
  crc = crc32(crc, crcbuf, outputModuleLabel.length());

  uint32 adler32_chksum = (uint32)cms::Adler32((char*)&test_value[0], sizeof(test_value));
  std::string host_name = "mytestnode.cms";

  InitMsgBuilder init(&buf[0],buf.size(),12,
                      Version((const uint8*)psetid),(const char*)reltag,
		      processName.c_str(),outputModuleLabel.c_str(), crc,
                      hlt_names,hlt_names,l1_names,
                      adler32_chksum, host_name.c_str());


  init.setDataLength(sizeof(test_value));
  std::copy(&test_value[0],&test_value[0]+sizeof(test_value),
            init.dataAddress());

  InitMsgView view(&buf[0]);
  uint8 psetid2[16];
  Strings hlt2;
  Strings l12;
  view.pset(psetid2);
  view.hltTriggerNames(hlt2);
  view.l1TriggerNames(l12);

  uint32 adler32_2 = view.adler32_chksum();
  std::string host_name2 = view.hostName();


  InitMsgBuilder init2(&buf2[0],buf2.size(),
                       view.run(),
                       Version((const uint8*)psetid2),
                       view.releaseTag().c_str(),
                       processName.c_str(),outputModuleLabel.c_str(), crc,
                       hlt2,hlt2,l12,
                       adler32_2,
                       host_name2.c_str());

  init2.setDataLength(view.descLength());
  std::copy(view.descData(),view.descData()+view.size(),
            init2.dataAddress());

  if(equal(&buf[0],&buf[0]+view.size(),buf2.begin())==false)
    {
      std::cerr << "Init buffers not the same!\n";
      try {
        dumpInit(&buf[0]);
        dumpInit(&buf2[0]);
      } catch(cms::Exception const&) {
      }
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

  adler32_chksum = (uint32)cms::Adler32((char*)&test_value[0], sizeof(test_value));
  //host_name = "mytestnode.cms";

  EventMsgBuilder emb(&buf[0],buf.size(),45,2020,2,0xdeadbeef,3,
                      l1bit,hltbits,hltsize, adler32_chksum, host_name.c_str());

  emb.setOrigDataSize(78);
  emb.setEventLength(sizeof(test_value));
  std::copy(&test_value[0],&test_value[0]+sizeof(test_value),
            emb.eventAddr());

  // ------ check for sameness

  EventMsgView eview(&buf[0]);

  std::vector<bool> l1_out;
  uint8 hlt_out[10];
  eview.l1TriggerBits(l1_out);
  eview.hltTriggerBits(hlt_out);
  adler32_2 = eview.adler32_chksum();
  host_name2 = eview.hostName();

  EventMsgBuilder emb2(&buf2[0],buf.size(),
                       eview.run(),
                       eview.event(),
                       eview.lumi(),
                       eview.outModId(),
                       eview.droppedEventsCount(),
                       l1_out,
                       hlt_out,
                       hltsize,
                       adler32_2,
                       host_name2.c_str());

  emb2.setOrigDataSize(eview.origDataSize());
  emb2.setEventLength(eview.eventLength());
  std::copy(eview.eventData(),eview.eventData()+eview.eventLength(),
            emb2.eventAddr());

  if(equal(&buf[0],&buf[0]+emb.size(),buf2.begin())==false)
    {
      std::cerr << "event messages not the same\n";
      std::cerr << "size 1st=" << emb.size() << "\n"
           << "size 2nd=" << emb2.size() << "\n";

      dumpEvent(&buf[0]);
      dumpEvent(&buf2[0]);
      abort();
    }
  return 0;
}


