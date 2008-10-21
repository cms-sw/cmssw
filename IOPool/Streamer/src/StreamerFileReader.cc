#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/src/StreamerFileReader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"

using namespace edm;

namespace edm
{  
  StreamerFileReader::StreamerFileReader(edm::ParameterSet const& pset):
    streamerNames_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"))
  {
      PoolCatalog poolcat;
      InputFileCatalog catalog(pset, poolcat);
      streamerNames_ = catalog.fileNames();

      if (streamerNames_.size() > 1)
          stream_reader_ = std::auto_ptr<StreamerInputFile> 
                          (new StreamerInputFile(streamerNames_));
      else if (streamerNames_.size() == 1) 
           stream_reader_ = std::auto_ptr<StreamerInputFile>
                          (new StreamerInputFile(streamerNames_.at(0)));
      else {
           throw edm::Exception(errors::FileReadError, "StreamerFileReader::StreamerFileReader")
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
      throw edm::Exception(errors::FileReadError, "StreamerFileReader::readHeader")
        << "received wrong message type: expected INIT, got "
        << header->code() << "\n";

   return header;
  }

 const EventMsgView* StreamerFileReader::getNextEvent()
 {
    if (!stream_reader_->next() )
    {
        return 0;
    }
    return stream_reader_->currentRecord();
 } 

} //end-of-namespace


