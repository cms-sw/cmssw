/** File contains simple tools to dump Init and Event 
    Messages on screen.
*/

#include "IOPool/Streamer/interface/DumpTools.h"

void dumpInitHeader(const InitMsgView* view)
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

  //cout<<"headerSize::::::"<<view->headerSize()<<endl;
  //cout << "Size of INIT Header: "<< view->initHeaderSize() << endl; 
  //cout << "Size of EVENT Header: "<< view->eventHeaderSize() << endl;  

  cout<<"# of HLT Bits: "<<vhltnames.size()<<endl;
  cout << "\nHLT names = \n";
  copy(vhltnames.begin(),vhltnames.end(),ostream_iterator<string>(cout,"\n"));
  cout<<"# of L1 Bits: "<<vl1names.size()<<endl;
  cout << "\nL1 names = \n";
  copy(vl1names.begin(),vl1names.end(),ostream_iterator<string>(cout,"\n"));
  cout << "\n";

}

void dumpInitView(const InitMsgView* view)
{

  dumpInitHeader(view);
  cout << "desc len = " << view->descLength() << "\n";
  //const uint8* pos = view->descData();
  //copy(pos,pos+view->descLength(),ostream_iterator<uint8>(cout,""));
  //cout << "\n";

}

void dumpStartMsg(const InitMsgView* view)
{
  dumpInitHeader(view);
}

void dumpInit(uint8* buf)
{
  InitMsgView view(buf);
  dumpInitHeader(&view);
  
  cout << "desc len = " << view.descLength() << "\n";
  //const uint8* pos = view.descData();
  //copy(pos,pos+view.descLength(),ostream_iterator<uint8>(cout,""));
  //cout << "\n";
}

void printBits(unsigned char c){

        for (int i=7; i>=0; i--) {
            int bit = ((c >> i) & 1);
            cout << " "<<bit;
        }
}


void dumpEventHeader(const EventMsgView* eview)
{
  cout << "code=" << eview->code() << "\n"
       << "size=" << eview->size() << "\n"
       << "run=" << eview->run() << "\n"
       << "event=" << eview->event() << "\n"
       << "lumi=" << eview->lumi() << "\n"
       << "reserved=" << eview->reserved() << "\n"
       << "event length=" << eview->eventLength() << "\n";

  std::vector<bool> l1_out;
  std::vector<unsigned char> hlt_out;
  hlt_out.resize(1 + (eview->hltCount())/4);

  //uint8 hlt_out[100];
  eview->l1TriggerBits(l1_out);
  eview->hltTriggerBits(&hlt_out[0]);
  //printBits(hlt_out[0]);  

  cout << "\nl1 size= " << l1_out.size() << "\n l1 bits=\n";
  copy(l1_out.begin(),l1_out.end(),ostream_iterator<bool>(cout," "));
   
  unsigned int bytesForHLT = 1 + (eview->hltCount())/4;

  cout << "\nBytes used by hlt = " << bytesForHLT <<endl; 

  cout << "\nhlt bits=\n(";
  copy(&hlt_out[0],&hlt_out[0]+(bytesForHLT-1),ostream_iterator<char>(cout,""));
  cout << ")\n";
 }

void dumpEventView(const EventMsgView* eview)
  {
  dumpEventHeader(eview);
  //const uint8* edata = eview->eventData();
  //cout << "\nevent data=\n(";
  //copy(&edata[0],&edata[0]+eview->eventLength(),
  //     ostream_iterator<char>(cout,""));
  //cout << ")\n";

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
  //cout << "\nevent data=\n(";
  //copy(&edata[0],&edata[0]+eview.eventLength(),
  //     ostream_iterator<char>(cout,""));
  //cout << ")\n";

}

