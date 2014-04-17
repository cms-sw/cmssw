#include <zlib.h>

#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/FEDInterface/interface/fed_header.h"
#include "EventFilter/FEDInterface/interface/fed_trailer.h"

#include "EventFilter/Utilities/plugins/FRDStreamSource.h"


FRDStreamSource::FRDStreamSource(edm::ParameterSet const& pset,
                                          edm::InputSourceDescription const& desc)
  : ProducerSourceFromFiles(pset,desc,true),
    verifyAdler32_(pset.getUntrackedParameter<bool> ("verifyAdler32", true))
{
  itFileName_=fileNames().begin();
  openFile(*itFileName_);
  produces<FEDRawDataCollection>();
}


bool FRDStreamSource::setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& theTime)
{
  if ( fin_.peek() == EOF ) {
    if ( ++itFileName_==fileNames().end() ) {
      fin_.close();
      return false;
    }
    if ( ! openFile(*itFileName_) ) {
      throw cms::Exception("FRDStreamSource::setRunAndEventInfo") <<
        "could not open file " << *itFileName_;
    }
  }

  const uint32_t headerSize = (4 + 1024) * sizeof(uint32_t); //minimal size to fit any version of FRDEventHeader
  if ( buffer_.size() < headerSize )
    buffer_.resize(headerSize);
  fin_.read(&buffer_[0],headerSize);

  // do we have to handle the case that a smaller header version + payload could fit into max headerSize?
  assert( fin_.gcount() == headerSize );

  std::unique_ptr<FRDEventMsgView> frdEventMsg(new FRDEventMsgView(&buffer_[0]));
  id = edm::EventID(frdEventMsg->run(), frdEventMsg->lumi(), frdEventMsg->event());

  const uint32_t totalSize = frdEventMsg->size();
  if ( totalSize > buffer_.size() ) {
    buffer_.resize(totalSize);
  }
  if ( totalSize > headerSize ) {
    fin_.read(&buffer_[0]+headerSize,totalSize-headerSize);
    if ( fin_.gcount() != totalSize-headerSize ) {
      throw cms::Exception("FRDStreamSource::setRunAndEventInfo") <<
        "premature end of file " << *itFileName_;
    }
    frdEventMsg.reset(new FRDEventMsgView(&buffer_[0]));
  }

  if ( verifyAdler32_ && frdEventMsg->version() >= 3 )
  {
    uint32_t adler = adler32(0L,Z_NULL,0);
    adler = adler32(adler,(Bytef*)frdEventMsg->payload(),frdEventMsg->eventSize());

    if ( adler != frdEventMsg->adler32() ) {
      throw cms::Exception("FRDStreamSource::setRunAndEventInfo") <<
        "Found a wrong Adler32 checksum: expected 0x" << std::hex << frdEventMsg->adler32() <<
        " but calculated 0x" << adler;
    }
  }

  rawData_.reset(new FEDRawDataCollection());

  uint32_t eventSize = frdEventMsg->eventSize();
  char* event = (char*)frdEventMsg->payload();

  while (eventSize > 0) {
    eventSize -= sizeof(fedt_t);
    const fedt_t* fedTrailer = (fedt_t*) (event + eventSize);
    const uint32_t fedSize = FED_EVSZ_EXTRACT(fedTrailer->eventsize) << 3; //trailer length counts in 8 bytes
    eventSize -= (fedSize - sizeof(fedh_t));
    const fedh_t* fedHeader = (fedh_t *) (event + eventSize);
    const uint16_t fedId = FED_SOID_EXTRACT(fedHeader->sourceid);
    if (fedId == FEDNumbering::MINTriggerGTPFEDID) {
      evf::evtn::evm_board_setformat(fedSize);
      const uint64_t gpsl = evf::evtn::getgpslow((unsigned char*) fedHeader);
      const uint64_t gpsh = evf::evtn::getgpshigh((unsigned char*) fedHeader);
      theTime = static_cast<edm::TimeValue_t>((gpsh << 32) + gpsl);
    }
    FEDRawData& fedData = rawData_->FEDData(fedId);
    fedData.resize(fedSize);
    memcpy(fedData.data(), event + eventSize, fedSize);
  }
  assert(eventSize == 0);

  return true;
}


void FRDStreamSource::produce(edm::Event& e) {
  e.put(rawData_);
}


bool FRDStreamSource::openFile(const std::string& fileName)
{
  fin_.close();
  fin_.clear();
  size_t pos = fileName.find(':');
  if (pos!=std::string::npos) {
    std::string prefix = fileName.substr(0,pos);
    if (prefix!="file") return false;
    pos++;
  }
  else pos=0;

  fin_.open(fileName.substr(pos).c_str(),std::ios::in|std::ios::binary);
  return fin_.is_open();
}


//////////////////////////////////////////
// define this class as an input source //
//////////////////////////////////////////
DEFINE_FWK_INPUT_SOURCE(FRDStreamSource);

// Keep old naming from DAQ1
using ErrorStreamSource = FRDStreamSource;
DEFINE_FWK_INPUT_SOURCE(ErrorStreamSource);
