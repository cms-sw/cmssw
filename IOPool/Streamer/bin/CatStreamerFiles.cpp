#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/DumpTools.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "IOPool/Streamer/interface/StreamerOutputFile.h"

// Utility to concatenate streamer files outside of the framework
// Mimics the behavior of DAQ
// Largely copied from DiagStreamerFile
using namespace edm::streamer;

void help();
void mergefile(StreamerOutputFile&, std::string const&, bool);

//==========================================================================
int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "No command line argument supplied\n";
    help();
    return 1;
  }

  std::string outfile(argv[1]);
  std::vector<std::string> streamfiles(argv + 2, argv + argc);

  StreamerOutputFile stream_output(outfile);
  bool first = true;
  for (auto const& sf : streamfiles) {
    mergefile(stream_output, sf, first);
    first = false;
  }

  return 0;
}

//==========================================================================
void help() {
  std::cout << "Usage: CatStreamerFile output_file_name streamer_file_name"
            << " [streamer_file_name ...]" << std::endl;
}

//==========================================================================
void mergefile(StreamerOutputFile& stream_output, std::string const& filename, bool includeheader) {
  uint32 num_events(0);

  try {
    StreamerInputFile stream_reader(filename);

    std::cout << "Trying to Read The Init message from Streamer File: " << std::endl << filename << std::endl;
    InitMsgView const* init = stream_reader.startMessage();
    if (includeheader) {
      std::cout << "Writing Init message to output file" << std::endl;
      stream_output.write(*init);
    }

    std::cout << "Trying to read the Event messages" << std::endl;
    EventMsgView const* eview(nullptr);
    while (StreamerInputFile::Next::kEvent == stream_reader.next()) {
      eview = stream_reader.currentRecord();
      ++num_events;
      stream_output.write(*eview);
    }
    std::cout << "Finished processing " << num_events << " events" << std::endl;
  } catch (cms::Exception& e) {
    std::cerr << "Exception caught:  " << e.what() << std::endl
              << "After reading " << num_events << "from file " << filename << std::endl;
  }
}
