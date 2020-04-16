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
       is handled. Basically StreamerInputFile(reader) has newHeader()
       returning TRUE, only ONCE when a New file is opened an a INIT Message
       is read instead of an Event message during next().

  main():

      Code entry point, comment the function call that you don't want to make.

*/

#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/DumpTools.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include <iostream>

int readSingleStream(bool verbose) {
  try {
    // ----------- init
    std::string initfilename = "teststreamfile.dat";
    edm::StreamerInputFile stream_reader(initfilename);

    std::cout << "Trying to Read The Init message from Streamer File: " << initfilename << std::endl;
    InitMsgView const* init = stream_reader.startMessage();
    if (verbose) {
      std::cout << "\n\n-------------INIT---------------------" << std::endl;
      std::cout << "Dump the Init Message from Streamer:-" << std::endl;
      dumpInitView(init);
    }

    // ------- event

    while (edm::StreamerInputFile::Next::kEvent == stream_reader.next()) {
      EventMsgView const* eview = stream_reader.currentRecord();
      if (verbose) {
        std::cout << "----------EVENT-----------" << std::endl;
        dumpEventView(eview);
      }
    }

  } catch (cms::Exception& e) {
    std::cerr << "Exception caught:  " << e.what() << std::endl;
    return 1;
  }
  return 0;
}

int readMultipleStreams(bool verbose) {
  try {
    std::unique_ptr<edm::SiteLocalConfig> slcptr =
        std::make_unique<edm::service::SiteLocalConfigService>(edm::ParameterSet());
    auto slc = std::make_shared<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> >(std::move(slcptr));
    edm::ServiceToken slcToken = edm::ServiceRegistry::createContaining(slc);
    edm::ServiceRegistry::Operate operate(slcToken);

    int evCount = 0;
    std::vector<std::string> streamFiles;
    streamFiles.push_back("file:teststreamfile.dat");
    streamFiles.push_back("file:teststreamfile.dat");

    edm::InputFileCatalog catalog(streamFiles, "");

    edm::StreamerInputFile stream_reader(catalog.fileCatalogItems());

    std::cout << "Trying to Read The Init message from Streamer File: "
              << "teststreamfile.dat" << std::endl;

    InitMsgView const* init = stream_reader.startMessage();
    if (verbose) {
      std::cout << "\n\n-------------INIT---------------------" << std::endl;
      std::cout << "Dump the Init Message from Streamer:-" << std::endl;
      dumpInitView(init);
    }

    while (edm::StreamerInputFile::Next::kStop != stream_reader.next()) {
      if (stream_reader.newHeader()) {
        std::cout << "File Boundary has just been crossed, a new file is read" << std::endl;
        std::cout << "A new INIT Message is available" << std::endl;
        std::cout << "Event from next file is also avialble" << std::endl;
        stream_reader.openNextFile();
        continue;
      }
      EventMsgView const* eview = stream_reader.currentRecord();
      if (verbose) {
        std::cout << "----------EVENT-----------" << std::endl;
        dumpEventView(eview);
      }
      ++evCount;
    }

    std::cout << " TOTAL Events Read: " << evCount << std::endl;
  } catch (cms::Exception& e) {
    std::cerr << "Exception caught:  " << e.what() << std::endl;
    return 1;
  }
  return 0;
}

int readInvalidLFN(bool verbose) {
  try {
    std::unique_ptr<edm::SiteLocalConfig> slcptr =
        std::make_unique<edm::service::SiteLocalConfigService>(edm::ParameterSet());
    auto slc = std::make_shared<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> >(std::move(slcptr));
    edm::ServiceToken slcToken = edm::ServiceRegistry::createContaining(slc);
    edm::ServiceRegistry::Operate operate(slcToken);

    int evCount = 0;
    std::vector<std::string> streamFiles;
    streamFiles.push_back("teststreamfile.dat");

    edm::InputFileCatalog catalog(streamFiles, "");

    edm::StreamerInputFile stream_reader(catalog.fileCatalogItems());

    std::cout << "Trying to Read The Init message from Streamer File: "
              << "teststreamfile.dat" << std::endl;

    InitMsgView const* init = stream_reader.startMessage();
    if (verbose) {
      std::cout << "\n\n-------------INIT---------------------" << std::endl;
      std::cout << "Dump the Init Message from Streamer:-" << std::endl;
      dumpInitView(init);
    }

    while (edm::StreamerInputFile::Next::kStop != stream_reader.next()) {
      if (stream_reader.newHeader()) {
        std::cout << "File Boundary has just been crossed, a new file is read" << std::endl;
        std::cout << "A new INIT Message is available" << std::endl;
        std::cout << "Event from next file is also avialble" << std::endl;
        stream_reader.openNextFile();
        continue;
      }
      EventMsgView const* eview = stream_reader.currentRecord();
      if (verbose) {
        std::cout << "----------EVENT-----------" << std::endl;
        dumpEventView(eview);
      }
      ++evCount;
    }

    std::cout << " TOTAL Events Read: " << evCount << std::endl;
  } catch (cms::Exception& e) {
    std::cerr << "Exception caught:  " << e.what() << std::endl;
    if (e.category() == "LogicalFileNameNotFound")
      return 0;
  }
  return 1;
}

void help() {
  std::cout << "Valid options are: " << std::endl;
  std::cout << "single, multi, invalid, all" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "No command line argument supplied\n";
    help();
    return 1;
  }

  std::string doThis(argv[1]);

  int ret(0);
  if (doThis == "all" || doThis == "single")
    ret += readSingleStream(false);
  if (doThis == "all" || doThis == "multi")
    ret += readMultipleStreams(false);
  if (doThis == "all" || doThis == "invalid")
    ret += readInvalidLFN(false);
  std::cout << "\n\nReadStreamerFile TEST DONE\n" << std::endl;

  return ret;
}
