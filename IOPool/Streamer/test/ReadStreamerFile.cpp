/** Sample code to Read Streammer and Index files in several possible scenarios

Disclaimer: Most of the code here is randomly written during
               testing various parts, its not a supported testing code.
               Changes can and will be made, when and if required.

   following functions and scenarios:

   useIndexPtr():

       Creates an Object of StreamerInputIndexFile pass it to create
       Object of StreamerInputFile,
       Using CTOR:
              StreamerInputFile(string const& name, string const& order);
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

void useIndexPtr() {
try{
  // edm::StreamerInputFile can make use of a index reference,
  // if you already have one
  //
  std::string initfilename = "teststreamfile.dat";
  std::string indexfilename = "testindexfile.ind";

  edm::StreamerInputIndexFile indexer(indexfilename);
  std::cout<<"indexer: "<< std::endl;
  edm::StreamerInputFile streamer(initfilename, indexer);

  // ------- events
  while(streamer.next()) {
     std::cout<<"----------EVENT-----------"<< std::endl;
     EventMsgView const* eview = streamer.currentRecord();
     dumpEventView(eview);
  }

  /******
      // Uncomment this block only if you are dieing to see
      // Start and Init Messages, Other pieces of code in
      // this file also does that so why repeat.

  // Start from Index file

  std::cout<<"\n\n-------------START---------------------"<< std::endl;
  StartIndexRecord* startindx = (StartIndexRecord*)
                                 indexer.startMessage();
  std::cout<<"Dump the Start Message from Index:-"<< std::endl;
  std::cout<<"Magic Number is: "<<startindx->magic<< std::endl;
  std::cout<<"Reserved filed is: "<<startindx->reserved<< std::endl;
  InitMsgView* start = (InitMsgView*) startindx->init;
  dumpStartMsg(start);

  //--------- init
  std::cout << "Trying to Read The Init message from Streamer File: "
       << initfilename << std::endl;
  InitMsgView* init =  streamer.startMessage();
  std::cout<<"\n\n-------------INIT---------------------"<< std::endl;
  std::cout<<"Dump the Init Message from Streamer:-"<< std::endl;
  dumpInitView(init);
  ******/
}catch (cms::Exception& e){
   std::cerr << "Exception caught:  "
             << e.what()
             << std::endl;
}

}

void getIndexPtr() {
try {
  // edm::StreamerInputFile Opens the Index file for you too !.
  // This test shows that you can access the index as well.
  // Though may not be very useful.

  std::string initfilename = "teststreamfile.dat";
  std::string indexfilename = "testindexfile.ind";
  edm::StreamerInputFile readViaIndex(initfilename, indexfilename);

  edm::StreamerInputIndexFile* indexPtr = (edm::StreamerInputIndexFile*)readViaIndex.index();

  // ------- event index
  for(indexRecIter it = indexPtr->begin(), itEnd = indexPtr->end(); it != itEnd; ++it) {
    std::cout<<"----------EVENT  INDEX-----------"<< std::endl;
    EventMsgView const* iview = (*it)->getEventView();
    dumpEventIndex(iview);
    std::cout << "Offset for this event is : "
         << (*it)->getOffset() << std::endl;
    }
} catch (cms::Exception& e){
   std::cerr << "Exception caught:  "
             << e.what()
             << std::endl;
}

}

void viaIndex() {


  /** Opens Streamer and corresponding Index file and then
      Loops over the events */
try{
  std::string initfilename = "teststreamfile.dat";
  std::string indexfilename = "testindexfile.ind";
  edm::StreamerInputFile readViaIndex(initfilename, indexfilename);

  // Dump events
  while(readViaIndex.next()) {
     std::cout<<"----------EVENT-----------"<< std::endl;
     EventMsgView const* eview = readViaIndex.currentRecord();
     dumpEventView(eview);
  }

  /******
      // Uncomment this block only if you are dieing to see
      // Start and Init Messages, Other pieces of code in
      // this file also does that so why repeat.

  //--------- init
  std::cout << "Trying to Read The Init message from Streamer File: "
       << initfilename << std::endl;
  InitMsgView const* init = readViaIndex.startMessage();
  std::cout<<"\n\n-------------INIT---------------------"<< std::endl;
  std::cout<<"Dump the Init Message from Streamer:-"<< std::endl;
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
  std::string initfilename = "teststreamfile.dat";
  edm::StreamerInputFile stream_reader (initfilename);

  std::cout << "Trying to Read The Init message from Streamer File: "
       << initfilename << std::endl;
  InitMsgView const* init = stream_reader.startMessage();
  std::cout<<"\n\n-------------INIT---------------------"<< std::endl;
  std::cout<<"Dump the Init Message from Streamer:-"<< std::endl;
  dumpInitView(init);

  // ----------- start

  std::cout<<"\n\n-------------START---------------------"<< std::endl;
  std::string indexfilename = "testindexfile.ind";

  /****
     // Uncomment this block and comment next one line
     // (Single file constructor) if you need
     //  to try out Multi file constructor for index files.
     // edm::StreamerInputIndexFile(std::vector<std::string> const& names);

  std::vector<sd::string> indexfiles;
  indexfiles.push_back(indexfilename);
  indexfiles.push_back(indexfilename);
  edm::StreamerInputIndexFile index_reader(indexfiles);
  *****/

  // ------- event

  while(stream_reader.next()) {
     std::cout<<"----------EVENT-----------"<< std::endl;
     EventMsgView const* eview = stream_reader.currentRecord();
     dumpEventView(eview);
  }

  //Index
  //Single file constructor
  edm::StreamerInputIndexFile index_reader(indexfilename);

  StartIndexRecord const* startindx = index_reader.startMessage();
  std::cout <<"---------------------Start Message-------------"<< std::endl;
  std::cout<<"Dump the Start Message from Index:-"<< std::endl;
  std::cout<<"Magic Number is: "<<startindx->getMagic()<< std::endl;
  std::cout<<"Reserved filed is: "<<startindx->getReserved()<< std::endl;
  InitMsgView const* start = startindx->getInit();
  dumpStartMsg(start);

  // ------- event index
  for(indexRecIter it = index_reader.begin(), itEnd = index_reader.end(); it!= itEnd; ++it) {
    std::cout<<"----------EVENT  INDEX-----------"<< std::endl;
    EventMsgView const* iview = (*it)->getEventView();
    dumpEventIndex(iview);
    std::cout << "Offset for this event is : "
         << (*it)->getOffset() << std::endl;
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
  std::vector<std::string> streamFiles;
  streamFiles.push_back("teststreamfile0.dat");
  streamFiles.push_back("teststreamfile1.dat");

  edm::StreamerInputFile stream_reader(streamFiles);

  std::cout << "Trying to Read The Init message from Streamer File: "
       << "teststreamfile0.dat" << std::endl;

  InitMsgView const* init = stream_reader.startMessage();
  std::cout<<"\n\n-------------INIT---------------------"<< std::endl;
  std::cout<<"Dump the Init Message from Streamer:-"<< std::endl;
  dumpInitView(init);

  while(stream_reader.next()) {
     if (stream_reader.newHeader()) {
           std::cout << "File Boundary has just been crossed, a new file is read" << std::endl;
           std::cout << "A new INIT Message is available" << std::endl;
           std::cout << "Event from next file is also avialble" << std::endl;
     }
     std::cout << "----------EVENT-----------" << std::endl;
     EventMsgView const* eview = stream_reader.currentRecord();
     dumpEventView(eview);
     ++evCount;
  }

 std::cout <<" TOTAL Events Read: "<<evCount<< std::endl;
}catch (cms::Exception& e){
   std::cerr << "Exception caught:  "
             << e.what()
             << std::endl;
   return 1;
}
return 0;
}

void help() {
      std::cout << "Valid options are: " << std::endl;
      std::cout<<"sep, via, gidx, indx, multi, all"<< std::endl;
}

int main(int argc, char* argv[]){

   if (argc < 2)
   {
      std::cout << "No command line argument supplied\n";
      help();
      return 0;
   }

   std::string doThis(argv[1]);

   if(doThis == "all" || doThis == "sep") separetly();
   if(doThis == "all" || doThis == "via") viaIndex();
   if(doThis == "all" || doThis == "gidx") getIndexPtr();
   if(doThis == "all" || doThis == "indx") useIndexPtr();
   if(doThis == "all" || doThis == "multi") readMultipleStreams();
   std::cout <<"\n\nReadStreamerFile TEST DONE\n"<< std::endl;

   return 0;
}

