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

  readMultipleStreams():
       Reads multiple stream files and iterates through all events.
       The test case also show, how the File boundary crossing "event"
       is handled. Basically StreamerInputFile (reader) has newHeader()
       returning TRUE, only ONCE when a New file is opened an a INIT Message
       is read instead of an Event message during next().

  main():

      Code entry point, comment the function call that you don't want to make.

*/

#include <iostream>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/DumpTools.h"

#include "IOPool/Streamer/interface/StreamerInputIndexFile.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "IOPool/Streamer/interface/IndexRecords.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;


void useIndexPtr() {
try{
  // StreamerInputFile can make use of a index reference,
  // if you already have one
  //
  string initfilename = "teststreamfile.dat";
  string indexfilename = "testindexfile.ind";

  StreamerInputIndexFile indexer(indexfilename);
  cout<<"indexer: "<<endl;
  StreamerInputFile streamer(initfilename, indexer);

  // ------- events
  while(streamer.next()) {
     cout<<"----------EVENT-----------"<<endl;
     const EventMsgView* eview = streamer.currentRecord();
     dumpEventView(eview);
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
  InitMsgView* init =  streamer.startMessage();
  cout<<"\n\n-------------INIT---------------------"<<endl;
  cout<<"Dump the Init Message from Streamer:-"<<endl;
  dumpInitView(init);
  ******/
}catch (cms::Exception& e){
   std::cerr << "Exception caught:  "
             << e.what()
             << std::endl;
}

}

void getIndexPtr() {
try{
  // StreamerInputFile Opens the Index file for you too !.
  // This test shows that you can access the index as well.
  // Though may not be very useful.

  string initfilename = "teststreamfile.dat";
  string indexfilename = "testindexfile.ind";
  StreamerInputFile readViaIndex(initfilename, indexfilename);

  StreamerInputIndexFile* indexPtr = (StreamerInputIndexFile*)readViaIndex.index();

  // ------- event index
  for(indexRecIter it = indexPtr->begin(), itEnd = indexPtr->end(); it != itEnd; ++it) {
    cout<<"----------EVENT  INDEX-----------"<<endl;
    const EventMsgView* iview = (*it)->getEventView();
    dumpEventIndex(iview);
    cout << "Offset for this event is : "
         << (*it)->getOffset() << endl;
    }
}catch (cms::Exception& e){
   std::cerr << "Exception caught:  "
             << e.what()
             << std::endl;
}

}

void viaIndex() {


  /** Opens Streamer and corresponding Index file and then
      Loops over the events */
try{
  string initfilename = "teststreamfile.dat";
  string indexfilename = "testindexfile.ind";
  StreamerInputFile readViaIndex(initfilename, indexfilename);

  // Dump events
  while(readViaIndex.next()) {
     cout<<"----------EVENT-----------"<<endl;
     const EventMsgView* eview = readViaIndex.currentRecord();
     dumpEventView(eview);
  }

  /******
      // Uncomment this block only if you are dieing to see
      // Start and Init Messages, Other pieces of code in
      // this file also does that so why repeat.

  //--------- init
  cout << "Trying to Read The Init message from Streamer File: "
       << initfilename << endl;
  const InitMsgView* init = readViaIndex.startMessage();
  cout<<"\n\n-------------INIT---------------------"<<endl;
  cout<<"Dump the Init Message from Streamer:-"<<endl;
  dumpInitView(init);

  ******/
}catch (cms::Exception& e){
   std::cerr << "Exception caught:  "
             << e.what()
             << std::endl;
}
}

void separetly() {
try{
  // ----------- init
  string initfilename = "teststreamfile.dat";
  StreamerInputFile stream_reader (initfilename);

  cout << "Trying to Read The Init message from Streamer File: "
       << initfilename << endl;
  const InitMsgView* init = stream_reader.startMessage();
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

  // ------- event

  while(stream_reader.next()) {
     cout<<"----------EVENT-----------"<<endl;
     const EventMsgView* eview = stream_reader.currentRecord();
     dumpEventView(eview);
  }

  //Index
  //Single file constructor
  StreamerInputIndexFile index_reader(indexfilename);

  const StartIndexRecord* startindx = index_reader.startMessage();
  cout <<"---------------------Start Message-------------"<<endl;
  cout<<"Dump the Start Message from Index:-"<<endl;
  cout<<"Magic Number is: "<<startindx->getMagic()<<endl;
  cout<<"Reserved filed is: "<<startindx->getReserved()<<endl;
  const InitMsgView* start = startindx->getInit();
  dumpStartMsg(start);

  // ------- event index
  for(indexRecIter it = index_reader.begin(), itEnd = index_reader.end(); it!= itEnd; ++it) {
    cout<<"----------EVENT  INDEX-----------"<<endl;
    const EventMsgView* iview = (*it)->getEventView();
    dumpEventIndex(iview);
    cout << "Offset for this event is : "
         << (*it)->getOffset() << endl;
    }
}catch (cms::Exception& e){
   std::cerr << "Exception caught:  "
             << e.what()
             << std::endl;
}
}


int readMultipleStreams()
{
try{  int evCount=0;
  vector<string> streamFiles;
  streamFiles.push_back("teststreamfile0.dat");
  streamFiles.push_back("teststreamfile1.dat");

  StreamerInputFile stream_reader(streamFiles);

  cout << "Trying to Read The Init message from Streamer File: "
       << "teststreamfile0.dat" << endl;

  const InitMsgView* init = stream_reader.startMessage();
  cout<<"\n\n-------------INIT---------------------"<<endl;
  cout<<"Dump the Init Message from Streamer:-"<<endl;
  dumpInitView(init);

  while(stream_reader.next()) {
     if (stream_reader.newHeader()) {
           cout << "File Boundary has just been crossed, a new file is read" << endl;
           cout << "A new INIT Message is available" << endl;
           cout << "Event from next file is also avialble" << endl;
     }
     cout << "----------EVENT-----------" << endl;
     const EventMsgView* eview = stream_reader.currentRecord();
     dumpEventView(eview);
     ++evCount;
  }

 cout <<" TOTAL Events Read: "<<evCount<<endl;
}catch (cms::Exception& e){
   std::cerr << "Exception caught:  "
             << e.what()
             << std::endl;
   return 1;
}
return 0;
}

void help() {
      cout << "Valid options are: " <<endl;
      cout<<"sep, via, gidx, indx, multi, all"<<endl;
}

int main(int argc, char* argv[]){

   if (argc < 2)
   {
      cout << "No command line argument supplied\n";
      help();
      return 0;
   }

   string doThis(argv[1]);

   if(doThis == "all" || doThis == "sep") separetly();
   if(doThis == "all" || doThis == "via") viaIndex();
   if(doThis == "all" || doThis == "gidx") getIndexPtr();
   if(doThis == "all" || doThis == "indx") useIndexPtr();
   if(doThis == "all" || doThis == "multi") readMultipleStreams();
   cout <<"\n\nReadStreamerFile TEST DONE\n"<<endl;

   return 0;
}

