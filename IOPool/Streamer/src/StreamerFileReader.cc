#include "IOPool/Streamer/src/StreamerFileReader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

using namespace std;
using namespace edm;

namespace edm
{  
  StreamerFileReader::StreamerFileReader(edm::ParameterSet const& pset):
    streamerNames_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"))
  {
        if (streamerNames_.size() > 1)
           stream_reader_ = std::auto_ptr<StreamerInputFile> 
                          (new StreamerInputFile(streamerNames_));
        else if (streamerNames_.size() == 1) 
           stream_reader_ = std::auto_ptr<StreamerInputFile>
                          (new StreamerInputFile(streamerNames_.at(0)));
        else {
           throw cms::Exception("StreamerFileReader","StreamerFileReader")
              << " Not provided fileNames \n";
        }
  }

  StreamerFileReader::~StreamerFileReader()
  {
  }

  const bool StreamerFileReader::newHeader() {
       return stream_reader_->newHeader();
  }

  const InitMsgView* StreamerFileReader::getHeader()
  {
 
    const InitMsgView* header = stream_reader_->startMessage();
  
    if(header->code() != Header::INIT) //INIT Msg
      throw cms::Exception("readHeader","StreamerFileReader")
        << "received wrong message type: expected INIT, got "
        << header->code() << "\n";

   return header;
  }

 const EventMsgView* StreamerFileReader::getNextEvent()
 {
    if (! stream_reader_->next() )
    {
        return 0;
    }
    return stream_reader_->currentRecord();
 } 

} //end-of-namespace


