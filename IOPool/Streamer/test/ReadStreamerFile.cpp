#include <iostream>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "IOPool/Streamer/interface/StreamerFileIO.h"

#include "IOPool/Streamer/interface/DumpTools.h"

using namespace std;

int main()
{ 

  // ----------- init

  string initfilename = "teststreamfile.dat";
  StreamerInputFile stream_reader (initfilename);

  cout << "Trying to Read The Init message from Streamer File: " 
       << initfilename << endl;
  InitMsgView* init = (InitMsgView*) stream_reader.startMessage();
  cout<<"\n\n-------------INIT---------------------"<<endl; 
  cout<<"Dump the Init Message from Streamer:-"<<endl;
  dumpInitView(init);

  // ----------- start

  string indexfilename = "testindexfile.ind";
  StreamerInputIndexFile index_reader(indexfilename);
  StartIndexRecord* startindx = (StartIndexRecord*) 
                                 index_reader.startMessage(); 

  cout<<"\n\n-------------START---------------------"<<endl;
  cout<<"Dump the Start Message from Index:-"<<endl; 
  cout<<"Magic Number is: "<<startindx->magic<<endl;
  cout<<"Reserved filed is: "<<startindx->reserved<<endl;
  InitMsgView* start = (InitMsgView*) startindx->init;
  dumpStartMsg(start);
     
  // ------- event

  std::vector<bool> l1bit(11);
  uint8 hltbits[] = "4567";
  //const int hltsize = (sizeof(hltbits)-1)*4;
  const int hltsize = 9;
  //cout<<"....DUMMY SIZES here ......"<<endl;
  cout<<"hltsize: "<<hltsize<<endl;
  cout<<"l1bit.size(): "<<l1bit.size()<<endl;

  //return 0;


  while( stream_reader.next() ) {
     cout<<"----------EVENT-----------"<<endl;
     EventMsgView* eview = (EventMsgView*) stream_reader.currentRecord();
     dumpEventView(eview, hltsize, l1bit.size());
  }

  // ------- event index
 
  while( index_reader.next() ) {
     cout<<"----------EVENT  INDEX-----------"<<endl;
     //This should also work
     //EventMsgView* iview = (EventMsgView*) index_reader.currentRecord();
     //dumpEventView(iview, eview->size(), hltsize, l1bit.size());
     
     EventIndexRecord* iview = (EventIndexRecord*) 
                             index_reader.currentRecord();
     dumpEventIndex(iview->eview, hltsize, l1bit.size());
     cout<<"Offset for this event is : "<<iview->offset<<endl;      
     
  }

  return 0;
}


