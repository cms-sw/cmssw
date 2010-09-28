/** Sample code to Read Streammer files in several possible scenarios

Disclaimer: Most of the code here is randomly written during
            testing various parts, its not a supported testing code.
            Changes can and will be made, when and if required.

   following functions and scenarios:

  readSingleStream():
       Reads a single streamer file. It shows how the InitMsg and EventMsg
       can be accessed.

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

#include "IOPool/Streamer/interface/StreamerInputFile.h"

#include "FWCore/Utilities/interface/Exception.h"

void readSingleStream() {
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

  // ------- event

  while(stream_reader.next()) {
     std::cout<<"----------EVENT-----------"<< std::endl;
     EventMsgView const* eview = stream_reader.currentRecord();
     dumpEventView(eview);
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
      std::cout<<"single, multi, all"<< std::endl;
}

int main(int argc, char* argv[]){

   if (argc < 2)
   {
      std::cout << "No command line argument supplied\n";
      help();
      return 0;
   }

   std::string doThis(argv[1]);

   if(doThis == "all" || doThis == "single") readSingleStream();
   if(doThis == "all" || doThis == "multi") readMultipleStreams();
   std::cout <<"\n\nReadStreamerFile TEST DONE\n"<< std::endl;

   return 0;
}

