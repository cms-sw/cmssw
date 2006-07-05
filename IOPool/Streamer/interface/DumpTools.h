/** File contains simple tools to dump Init and Event 
    Messages on screen.
*/

#ifndef _dump_tool_
#define _dump_tool_

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

void dumpInitHeader(InitMsgView* view)
{
  cout
    << "code = " << view->code() << ", "
    << "size = " << view->size() << "\n"
    << "run = " << view->run() << ", "
    << "proto = " << view->protocolVersion() << "\n"
    << "release = " << view->releaseTag() << "\n";

  uint8 vpset[17];
  view->pset(vpset);
  vpset[16]='\0';
  Strings vhltnames,vl1names;
  view->hltTriggerNames(vhltnames);
  view->l1TriggerNames(vl1names);

  cout << "pset = " << vpset << "\n";
  cout << "\nHLT names = \n";


  cout<<"# of HLT Bits: "<<vhltnames.size()<<endl;
  copy(vhltnames.begin(),vhltnames.end(),ostream_iterator<string>(cout,"\n"));
  cout<<"# of L1 Bits: "<<vl1names.size()<<endl;
  cout << "\nL1 names = \n";
  copy(vl1names.begin(),vl1names.end(),ostream_iterator<string>(cout,"\n"));
  cout << "\n";

}

void dumpInitView(InitMsgView* view)
{

  dumpInitHeader(view);
  cout << "desc len = " << view->descLength() << "\n";
  const uint8* pos = view->descData();
  copy(pos,pos+view->descLength(),ostream_iterator<uint8>(cout,""));
  cout << "\n";

}

void dumpStartMsg(InitMsgView* view)
{
  dumpInitHeader(view);
}

void dumpInit(uint8* buf, uint32 bufsize)
{
  InitMsgView view(buf);
  dumpInitHeader(&view);
  
  cout << "desc len = " << view.descLength() << "\n";
  const uint8* pos = view.descData();
  copy(pos,pos+view.descLength(),ostream_iterator<uint8>(cout,""));
  cout << "\n";
}


void dumpEventHeader(EventMsgView* eview, uint32 hltsize,uint32 l1size)
{
  cout << "code=" << eview->code() << "\n"
       << "size=" << eview->size() << "\n"
       << "run=" << eview->run() << "\n"
       << "event=" << eview->event() << "\n"
       << "lumi=" << eview->lumi() << "\n"
       << "reserved=" << eview->reserved() << "\n"
       << "event length=" << eview->eventLength() << "\n";

  std::vector<bool> l1_out;
  uint8 hlt_out[10];
  eview->l1TriggerBits(l1_out);
  eview->hltTriggerBits(hlt_out);

  cout << "\nl1 size= " << l1_out.size() << " l1 bits=\n";
  copy(l1_out.begin(),l1_out.end(),ostream_iterator<bool>(cout," "));

  cout << "\nhlt bits=\n(";
  copy(&hlt_out[0],&hlt_out[0]+hltsize/4,ostream_iterator<char>(cout,""));
  cout << ")\n";
 }

void dumpEventView(EventMsgView* eview, uint32 hltsize,uint32 l1size)
  {
  dumpEventHeader(eview, hltsize, l1size);
  const uint8* edata = eview->eventData();
  cout << "\nevent data=\n(";
  copy(&edata[0],&edata[0]+eview->eventLength(),
       ostream_iterator<char>(cout,""));
  cout << ")\n";

}

void dumpEventIndex(EventMsgView* eview, uint32 hltsize,uint32 l1size)
{
  dumpEventHeader(eview, hltsize, l1size);
}

void dumpEvent(uint8* buf,uint32 bufsize,uint32 hltsize,uint32 l1size)
{
  EventMsgView eview(buf,hltsize,l1size);

  dumpEventHeader(&eview, hltsize, l1size);

  const uint8* edata = eview.eventData();
  cout << "\nevent data=\n(";
  copy(&edata[0],&edata[0]+eview.eventLength(),
       ostream_iterator<char>(cout,""));
  cout << ")\n";

}

#endif

