/** File contains simple tools to dump Init and Event 
    Messages on screen.
*/

#include "IOPool/Streamer/interface/DumpTools.h"
#include "FWCore/Utilities/interface/Digest.h"
#include <iostream>
#include <iterator>

void dumpInitHeader(const InitMsgView* view)
{
  std::cout
    << "code = " << view->code() << ", "
    << "size = " << view->size() << "\n"
    << "run = " << view->run() << ", "
    << "proto = " << view->protocolVersion() << "\n"
    << "release = " << view->releaseTag() << "\n"
    << "processName = " << view->processName() << "\n";


  //PSet 16 byte non-printable representation, stored in message.
  uint8 vpset[16];
  view->pset(vpset); 

  //Lets convert it to printable hex form
  vpset[16]='\0';
  std::string pset_str((char*) &vpset[0]);
  cms::Digest dig(pset_str);
  cms::MD5Result r1 = dig.digest();
  std::string hexy = r1.toString();
  std::cout << "PSetID= " << hexy << std::endl;

  Strings vhltnames,vl1names;
  view->hltTriggerNames(vhltnames);
  view->l1TriggerNames(vl1names);

  std::cout << "HLT names :- \n ";
  std::copy(vhltnames.begin(),vhltnames.end(),std::ostream_iterator<std::string>(std::cout,"\n"));

  std::cout << "L1 names :- \n ";
  std::copy(vl1names.begin(),vl1names.end(),std::ostream_iterator<std::string>(std::cout,"\n"));
  std::cout << "\n";

}

void dumpInitView(const InitMsgView* view)
{


  dumpInitHeader(view);
  std::cout << "desc len = " << view->descLength() << "\n";
  //const uint8* pos = view->descData();
  //std::copy(pos,pos+view->descLength(),std::ostream_iterator<uint8>(std::cout,""));
  //std::cout << "\n";

}

void dumpStartMsg(const InitMsgView* view)
{
  dumpInitHeader(view);
}

void dumpInit(uint8* buf)
{
  InitMsgView view(buf);
  dumpInitHeader(&view);
  
  std::cout << "desc len = " << view.descLength() << "\n";
  //const uint8* pos = view.descData();
  //std::copy(pos,pos+view.descLength(),std::ostream_iterator<uint8>(std::cout,""));
  //std::cout << "\n";
}

void printBits(unsigned char c){

        for (int i = 7; i >= 0; --i) {
            int bit = ((c >> i) & 1);
            std::cout << " "<<bit;
        }
}


void dumpEventHeader(const EventMsgView* eview)
{
  std::cout << "code=" << eview->code() << "\n"
       << "size=" << eview->size() << "\n"
       << "run=" << eview->run() << "\n"
       << "event=" << eview->event() << "\n"
       << "lumi=" << eview->lumi() << "\n"
       << "reserved=" << eview->reserved() << "\n"
       << "event length=" << eview->eventLength() << "\n";

  std::vector<bool> l1_out;
  std::vector<unsigned char> hlt_out;
  hlt_out.resize(1 + (eview->hltCount()-1)/4);

  //uint8 hlt_out[100];
  eview->l1TriggerBits(l1_out);
  eview->hltTriggerBits(&hlt_out[0]);
  //printBits(hlt_out[0]);  

  std::cout << "\nl1 size= " << l1_out.size() << "\n l1 bits=\n";
  std::copy(l1_out.begin(),l1_out.end(),std::ostream_iterator<bool>(std::cout," "));
  
  unsigned int bytesForHLT = eview->hltCount();   
  std::cout << "\nhlt Count:" << eview->hltCount() << std::endl;
  if (eview->hltCount() !=0)  bytesForHLT = 1 + (eview->hltCount()-1)/4;

  std::cout << "\nhlt bits=\n(";
  for(int i=(hlt_out.size()-1); i != -1 ; --i) 
    printBits(hlt_out[i]);
  //std::copy(&hlt_out[0],&hlt_out[0]+bytesForHLT,std::ostream_iterator<char>(std::cout,""));
  std::cout << ")\n";
 }

void dumpEventView(const EventMsgView* eview)
  {
  dumpEventHeader(eview);
  //const uint8* edata = eview->eventData();
  //std::cout << "\nevent data=\n(";
  //std::copy(&edata[0],&edata[0]+eview->eventLength(),
  //     std::ostream_iterator<char>(std::cout,""));
  //std::cout << ")\n";

}

void dumpEventIndex(const EventMsgView* eview)
{
  dumpEventHeader(eview);
}

void dumpEvent(uint8* buf)
{
  EventMsgView eview(buf);

  dumpEventHeader(&eview);

  //const uint8* edata = eview.eventData();
  //std::cout << "\nevent data=\n(";
  //std::copy(&edata[0],&edata[0]+eview.eventLength(),
  //     std::ostream_iterator<char>(std::cout,""));
  //std::cout << ")\n";

}

