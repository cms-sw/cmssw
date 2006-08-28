/** Sample code to Read Streammer and Index files in several possible scenarios

Disclaimer: Most of the code here is randomly written during 
               testing various parts, its not a supported testing code.
               Changes can and will be made, when and if required. 

   following functions and scenarios:

   useIndexPtr(): 

       Creates an Object of StreamerInputIndexFile pass it to create 
       Object of StreamerInputFile, 
       Using CTOR: 
              StreamerInputFile(const string& name, const string& order); 
       Uses StreamerInputFile::next() to Loop over available Events, 
       The Events are Indexed by the supplied Index.     
      
  getIndexPtr():
       Creates an Object of StreamerInputFile, Passing it names of
       both a Streamer file and an index file, and then loops over
       the *Event Indexes* using the INDEP PTR obtained using,

           StreamerInputIndexFile* index();
        
  viaIndex():
       Creates an Object of StreamerInputFile, Passing it names of 
       both a Streamer file and an index file, and then loops over 
       the Events using StreamerInputFile::next(). 
       Similar to first scenario.

  separetly():
       Creates Streamer and Index iterators separately, using StreamerInputFile 
       and StreamerInputIndexFile Classes and loop over them seperatly, to 
       show that these files can be accessed individually if so desired.

  main():

      Code entry point, comment the function call that you don't want to make.

*/

#include <iostream>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include "IOPool/Streamer/interface/DumpTools.h"
using namespace std;

void useIndexPtr() {

  // StreamerInputFile can make use of a index reference, 
  // if you already have one
  //

  std::vector<bool> l1bit(11);
  //uint8 hltbits[] = "4567";
  //const int hltsize = (sizeof(hltbits)-1)*4;
  const int hltsize = 9;

  string initfilename = "teststreamfile.dat";
  string indexfilename = "testindexfile.ind";

  StreamerInputIndexFile indexer(indexfilename);
  StreamerInputFile streamer(initfilename, indexer);

  // ------- events
  while( streamer.next() ) {
     cout<<"----------EVENT-----------"<<endl;
     EventMsgView* eview = (EventMsgView*) streamer.currentRecord();
     dumpEventView(eview);//, hltsize, l1bit.size());
  }

  /******  
      // Uncomment this block only if you are dieing to see
      // Start and Init Messages, Other pieces of code in 
      // this file also does that so why repeat.

  // Start from Index file 

  cout<<"\n\n-------------START---------------------"<<endl;
  StartIndexRecord* startindx = (StartIndexRecord*)
                                 indexer.startMessage();
  cout<<"Dump the Start Message from Index:-"<<endl;
  cout<<"Magic Number is: "<<startindx->magic<<endl;
  cout<<"Reserved filed is: "<<startindx->reserved<<endl;
  InitMsgView* start = (InitMsgView*) startindx->init;
  dumpStartMsg(start);
   
  //--------- init
  cout << "Trying to Read The Init message from Streamer File: "
       << initfilename << endl;
  InitMsgView* init = (InitMsgView*) streamer.startMessage();
  cout<<"\n\n-------------INIT---------------------"<<endl;
  cout<<"Dump the Init Message from Streamer:-"<<endl;
  dumpInitView(init);
  ******/
}

void getIndexPtr() {

  // StreamerInputFile Opens the Index file for you too !.
  // This test shows that you can access the index as well.
  // Though may not be very useful.

  string initfilename = "teststreamfile.dat";
  string indexfilename = "testindexfile.ind";
  StreamerInputFile readViaIndex(initfilename, indexfilename);
  
  StreamerInputIndexFile* indexPtr = readViaIndex.index();
    
  std::vector<bool> l1bit(11);
  //uint8 hltbits[] = "4567";
  //const int hltsize = (sizeof(hltbits)-1)*4;
  const int hltsize = 9;

  // ------- event index
  indexRecIter it;
  for(it = indexPtr->begin(); it != indexPtr->end(); ++it) {
    cout<<"----------EVENT  INDEX-----------"<<endl;
    EventMsgView* iview = (EventMsgView*) (*it).eview;
    dumpEventIndex(iview);//, hltsize, l1bit.size());
    cout << "Offset for this event is : "
         << *((*it).offset) << endl;
    }
}

void viaIndex() {
  
  /** Opens Streamer and corresponding Index file and then
      Loops over the events */

  std::vector<bool> l1bit(11);
  //uint8 hltbits[] = "4567";
  //const int hltsize = (sizeof(hltbits)-1)*4;
  const int hltsize = 9;

  string initfilename = "teststreamfile.dat";
  string indexfilename = "testindexfile.ind";
  StreamerInputFile readViaIndex(initfilename, indexfilename);

  // Dump events
  while( readViaIndex.next() ) {
     cout<<"----------EVENT-----------"<<endl;
     EventMsgView* eview = (EventMsgView*) readViaIndex.currentRecord();
     dumpEventView(eview);//, hltsize, l1bit.size());
  }

  /******  

      // Uncomment this block only if you are dieing to see
      // Start and Init Messages, Other pieces of code in
      // this file also does that so why repeat.

  //--------- init
  cout << "Trying to Read The Init message from Streamer File: "
       << initfilename << endl;
  InitMsgView* init = (InitMsgView*) readViaIndex.startMessage();
  cout<<"\n\n-------------INIT---------------------"<<endl;
  cout<<"Dump the Init Message from Streamer:-"<<endl;
  dumpInitView(init);

  ******/

}

void separetly() {

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

  cout<<"\n\n-------------START---------------------"<<endl;
  string indexfilename = "testindexfile.ind";

  /**** 
     // Uncomment this block and comment next one line 
     // (Single file constructor) if you need 
     //  to try out Multi file constructor for index files.
     // StreamerInputIndexFile(const vector<string>& names); 

  vector<string> indexfiles;
  indexfiles.push_back(indexfilename);
  indexfiles.push_back(indexfilename);
  StreamerInputIndexFile index_reader(indexfiles);
  *****/

  //Single file constructor
  StreamerInputIndexFile index_reader(indexfilename);

  StartIndexRecord* startindx = (StartIndexRecord*) 
                                 index_reader.startMessage(); 

  cout<<"Dump the Start Message from Index:-"<<endl; 
  cout<<"Magic Number is: "<<*startindx->magic<<endl;
  cout<<"Reserved filed is: "<<*startindx->reserved<<endl;
  InitMsgView* start = (InitMsgView*) startindx->init;
  dumpStartMsg(start);
  
  // ------- event
  std::vector<bool> l1bit(11);
  //uint8 hltbits[] = "4567";
  //const int hltsize = (sizeof(hltbits)-1)*4;
  const int hltsize = 9;

  while( stream_reader.next() ) {
     cout<<"----------EVENT-----------"<<endl;
     EventMsgView* eview = (EventMsgView*) stream_reader.currentRecord();
     dumpEventView(eview);//, hltsize, l1bit.size());
  }

  // ------- event index
  indexRecIter it;
  for(it = index_reader.begin(); it != index_reader.end(); ++it) {
    cout<<"----------EVENT  INDEX-----------"<<endl;
    EventMsgView* iview = (EventMsgView*) (*it).eview;
    dumpEventIndex(iview);//, hltsize, l1bit.size());
    cout << "Offset for this event is : "
         << *((*it).offset) << endl;
    }
}

int main() 
{
  separetly();
  viaIndex();
  getIndexPtr();  
  useIndexPtr();

  return 0;
}

