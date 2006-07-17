#include "IOPool/Streamer/src/StreamerFileReader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

using namespace std;
using namespace edm;

namespace edm
{  
  StreamerFileReader::StreamerFileReader(edm::ParameterSet const& pset):
    filename_(pset.getParameter<string>("fileName")),
    stream_reader_(new StreamerInputFile(filename_.c_str()))
  {

  }

  StreamerFileReader::~StreamerFileReader()
  {
      //delete stream_reader_;
  }

  std::auto_ptr<InitMsgView> StreamerFileReader::getHeader()
  {
 
  std::auto_ptr<InitMsgView> header ( (InitMsgView*) stream_reader_->startMessage() );
  
  if(header->code() != 0) //INIT Msg
      throw cms::Exception("readHeader","StreamerFileReader")
        << "received wrong message type: expected INIT, got "
        << header->code() << "\n";

   return header;
  }

 std::auto_ptr<EventMsgView> StreamerFileReader::getNextEvent()
 {
    if (! stream_reader_->next() )
    {
        //cerr << "\n\n\n LAST EVENT Read from Input file"<<endl;
        //Return an empty
        std::auto_ptr<EventMsgView> eview(0);
        return eview;
    }

    std::auto_ptr<EventMsgView> eview ( (EventMsgView*)stream_reader_->currentRecord() );
      
    return eview;
 } 

} //end-of-namespace


