#include <zlib.h>
#include <iostream>

#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/FEDInterface/interface/fed_header.h"
#include "EventFilter/FEDInterface/interface/fed_trailer.h"
#include "EventFilter/FEDInterface/interface/FED1024.h"

#include "EventFilter/Utilities/plugins/FRDStreamSource.h"
#include "EventFilter/Utilities/interface/crc32c.h"


FRDStreamSource::FRDStreamSource(edm::ParameterSet const& pset,
                                          edm::InputSourceDescription const& desc)
  : ProducerSourceFromFiles(pset,desc,true),
    verifyAdler32_(pset.getUntrackedParameter<bool> ("verifyAdler32", true)),
    verifyChecksum_(pset.getUntrackedParameter<bool> ("verifyChecksum", true)),
    useL1EventID_(pset.getUntrackedParameter<bool> ("useL1EventID", false))
{
  itFileName_=fileNames().begin();
  openFile(*itFileName_);
  produces<FEDRawDataCollection>();
}


bool FRDStreamSource::setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& theTime, edm::EventAuxiliary::ExperimentType& eType)
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

  if ( detectedFRDversion_==0) {
    fin_.read((char*)&detectedFRDversion_,sizeof(uint32_t));
    assert(detectedFRDversion_>0 && detectedFRDversion_<=5);
    if ( buffer_.size() < FRDHeaderVersionSize[detectedFRDversion_] )
      buffer_.resize(FRDHeaderVersionSize[detectedFRDversion_]);
    *((uint32_t*)(&buffer_[0]))=detectedFRDversion_;
    fin_.read(&buffer_[0] + sizeof(uint32_t),FRDHeaderVersionSize[detectedFRDversion_]-sizeof(uint32_t));
    assert( fin_.gcount() == FRDHeaderVersionSize[detectedFRDversion_]-(unsigned int)(sizeof(uint32_t) ));
  }
  else {
    if ( buffer_.size() < FRDHeaderVersionSize[detectedFRDversion_] )
      buffer_.resize(FRDHeaderVersionSize[detectedFRDversion_]);
    fin_.read(&buffer_[0],FRDHeaderVersionSize[detectedFRDversion_]);
    assert( fin_.gcount() == FRDHeaderVersionSize[detectedFRDversion_] );
  }

  std::unique_ptr<FRDEventMsgView> frdEventMsg(new FRDEventMsgView(&buffer_[0]));
  if (useL1EventID_)
    id = edm::EventID(frdEventMsg->run(), frdEventMsg->lumi(), frdEventMsg->event());

  const uint32_t totalSize = frdEventMsg->size();
  if ( totalSize > buffer_.size() ) {
    buffer_.resize(totalSize);
  }
  if ( totalSize > FRDHeaderVersionSize[detectedFRDversion_] ) {
    fin_.read(&buffer_[0]+FRDHeaderVersionSize[detectedFRDversion_],totalSize-FRDHeaderVersionSize[detectedFRDversion_]);
    if ( fin_.gcount() != totalSize-FRDHeaderVersionSize[detectedFRDversion_] ) {
      throw cms::Exception("FRDStreamSource::setRunAndEventInfo") <<
        "premature end of file " << *itFileName_;
    }
    frdEventMsg.reset(new FRDEventMsgView(&buffer_[0]));
  }

  if ( verifyChecksum_ && frdEventMsg->version() >= 5 )
  {
    uint32_t crc=0;
    crc = crc32c(crc,(const unsigned char*)frdEventMsg->payload(),frdEventMsg->eventSize());
    if ( crc != frdEventMsg->crc32c() ) {
      throw cms::Exception("FRDStreamSource::getNextEvent") <<
        "Found a wrong crc32c checksum: expected 0x" << std::hex << frdEventMsg->crc32c() <<
        " but calculated 0x" << crc;
    }
  }
  else if ( verifyAdler32_ && frdEventMsg->version() >= 3 )
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
  bool foundTCDSFED=false;
  bool foundGTPFED=false;


  while (eventSize > 0) {
    assert(eventSize>=sizeof(fedt_t));
    eventSize -= sizeof(fedt_t);
    const fedt_t* fedTrailer = (fedt_t*) (event + eventSize);
    const uint32_t fedSize = FED_EVSZ_EXTRACT(fedTrailer->eventsize) << 3; //trailer length counts in 8 bytes
    assert(eventSize>=fedSize - sizeof(fedt_t));
    eventSize -= (fedSize - sizeof(fedt_t));
    const fedh_t* fedHeader = (fedh_t *) (event + eventSize);
    const uint16_t fedId = FED_SOID_EXTRACT(fedHeader->sourceid);
    if (fedId>FEDNumbering::MAXFEDID)
    {
      throw cms::Exception("FedRawDataInputSource::fillFEDRawDataCollection") << "Out of range FED ID : " << fedId;
    }
    if (fedId == FEDNumbering::MINTCDSuTCAFEDID) {
      foundTCDSFED=true;
      evf::evtn::TCDSRecord record((unsigned char *)(event + eventSize ));
      id = edm::EventID(frdEventMsg->run(),record.getHeader().getData().header.lumiSection,
			record.getHeader().getData().header.eventNumber);
      eType = ((edm::EventAuxiliary::ExperimentType)FED_EVTY_EXTRACT(fedHeader->eventid));
      //evf::evtn::evm_board_setformat(fedSize);
      uint64_t gpsh = record.getBST().getBST().gpstimehigh;
      uint32_t gpsl = record.getBST().getBST().gpstimelow;
      theTime = static_cast<edm::TimeValue_t>((gpsh << 32) + gpsl);
    }

    if (fedId == FEDNumbering::MINTriggerGTPFEDID && !foundTCDSFED) {
      foundGTPFED=true;
      const bool GTPEvmBoardSense=evf::evtn::evm_board_sense((unsigned char*) fedHeader,fedSize);
      if (!useL1EventID_) {
        if (GTPEvmBoardSense)
          id = edm::EventID(frdEventMsg->run(), frdEventMsg->lumi(), evf::evtn::get((unsigned char*) fedHeader,true));
        else
          id = edm::EventID(frdEventMsg->run(), frdEventMsg->lumi(), evf::evtn::get((unsigned char*) fedHeader,false));
      }
      //evf::evtn::evm_board_setformat(fedSize);
      const uint64_t gpsl = evf::evtn::getgpslow((unsigned char*) fedHeader);
      const uint64_t gpsh = evf::evtn::getgpshigh((unsigned char*) fedHeader);
      theTime = static_cast<edm::TimeValue_t>((gpsh << 32) + gpsl);
    }



    //take event ID from GTPE FED
    if (fedId == FEDNumbering::MINTriggerEGTPFEDID && !foundGTPFED && !foundTCDSFED && !useL1EventID_) {
      if (evf::evtn::gtpe_board_sense((unsigned char*)fedHeader)) {
        id = edm::EventID(frdEventMsg->run(), frdEventMsg->lumi(), evf::evtn::gtpe_get((unsigned char*) fedHeader));
      }
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
  std::cout << " open file.. " << fileName << std::endl;
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
