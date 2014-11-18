// -*- C++ -*-
//
// Package:    EventFilter/L1TRawToDigi
// Class:      MP7BufferDumpToRaw
// 
/**\class Stage2InputPatternWriter Stage2InputPatternWriter.cc L1Trigger/L1TCalorimeter/plugins/Stage2InputPatternWriter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  James Brooke
//         Created:  Tue, 11 Mar 2014 14:55:45 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <boost/algorithm/string.hpp>

#include "EventFilter/L1TRawToDigi/interface/MP7FileReader.h"
#include "EventFilter/L1TRawToDigi/interface/MP7PacketReader.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"

//
// class declaration
//

namespace l1t {

class MP7BufferDumpToRaw : public edm::EDProducer {
public:
  explicit MP7BufferDumpToRaw(const edm::ParameterSet&);
  ~MP7BufferDumpToRaw();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  void fillBlocks(int iAmc);

  void formatRaw(FEDRawData& feddata, int bx, int evtId, int orbit);
  
  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------

  // file readers
  MP7FileReader rxFileReader_;
  MP7FileReader txFileReader_;
  unsigned rxIndex_;
  unsigned txIndex_;


  // packet reader (if needed)
  //  MP7PacketReader rxPacketReader_;
  //  MP7PacketReader txPacketReader_;

  // formatting parameters
  bool packetisedData_;
 
  // non packetised data parameters
  unsigned nFramesPerEvent_;

  // packetised data parameters


  // DAQ parameters
  int fedId_;
  int evType_;
  int fwVer_;
  int lenSlinkHeader_;  // in 8-bit words
  int lenSlinkTrailer_;
  int lenAMC13Header_;
  int lenAMC13Trailer_;
  int lenAMCHeader_;   
  int lenAMCTrailer_;   
  std::vector<int> rxBlockLength_;
  std::vector<int> txBlockLength_;  


  // the data
  std::vector<Block> blocks_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
  MP7BufferDumpToRaw::MP7BufferDumpToRaw(const edm::ParameterSet& iConfig) :
    rxFileReader_(iConfig.getUntrackedParameter<std::string>("rxFile", "rx_summar\
y.txt")),
    txFileReader_(iConfig.getUntrackedParameter<std::string>("txFile", "tx_summar\
y.txt")),
    rxIndex_(0),
    txIndex_(0),
    packetisedData_(iConfig.getUntrackedParameter<bool>("packetisedData", true)),
    nFramesPerEvent_(iConfig.getUntrackedParameter<unsigned>("nFramesPerEvent", 6)),
    fedId_(iConfig.getUntrackedParameter<int>("fedId", 1)),
    evType_(iConfig.getUntrackedParameter<int>("eventType", 1)),
    fwVer_(iConfig.getUntrackedParameter<int>("fwVersion", 1)),
    lenSlinkHeader_(iConfig.getUntrackedParameter<int>("lenSlinkHeader", 16)),
    lenSlinkTrailer_(iConfig.getUntrackedParameter<int>("lenSlinkTrailer", 16)),
    lenAMC13Header_(iConfig.getUntrackedParameter<int>("lenAMC13Header", 0)),
    lenAMC13Trailer_(iConfig.getUntrackedParameter<int>("lenAMC13Trailer", 0)),
    lenAMCHeader_(iConfig.getUntrackedParameter<int>("lenAMCHeader", 12)),
    lenAMCTrailer_(iConfig.getUntrackedParameter<int>("lenAMCTrailer", 8)),
    rxBlockLength_(iConfig.getUntrackedParameter< std::vector<int> >("rxBlockLength")),
    txBlockLength_(iConfig.getUntrackedParameter< std::vector<int> >("txBlockLength"))
{

  produces<FEDRawDataCollection>();

  // advance pointers for non packetised data
  if (!packetisedData_) {
    rxIndex_ += iConfig.getUntrackedParameter<int>("nFramesOffset", 0);
    txIndex_ += rxIndex_;
    txIndex_ += iConfig.getUntrackedParameter<int>("nFramesLatency", 0);
  }

}


MP7BufferDumpToRaw::~MP7BufferDumpToRaw()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
MP7BufferDumpToRaw::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // fill the block structure
  blocks_.clear();

  int iAmc=0;
  fillBlocks(iAmc);

  // create the collection
  std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 
  
  // retrieve the target buffer
  FEDRawData& feddata=rawColl->FEDData(fedId_);

  // fill the RAW data
  int bx = iEvent.bunchCrossing();
  int evtId = iEvent.id().event();
  long int orbit = iEvent.orbitNumber();
  
  formatRaw(feddata, bx, evtId, orbit);

  // put the collection in the event
  iEvent.put(rawColl);  

}



void
MP7BufferDumpToRaw::fillBlocks(int iAmc)
{

  // Rx blocks first
  for (unsigned i=0; i<rxBlockLength_.size(); ++i) {
    
    unsigned id   = i*2;
    unsigned size = rxBlockLength_.at(i);

    std::vector<uint32_t> data;
    for (unsigned iFrame=rxIndex_; iFrame<size; ++iFrame) {
      if (!packetisedData_) {
	data.push_back( rxFileReader_.get(iAmc).link(i).at(iFrame) );
      }
    }
    
    Block block(id, data);
    blocks_.push_back(block);
    
  }
  
  // then Tx blocks
  for (unsigned i=0; i<txBlockLength_.size(); ++i) {
    
    unsigned id   = (i*2)+1;
    unsigned size = txBlockLength_.at(i);

    std::vector<uint32_t> data(size);
    for (unsigned iFrame=txIndex_; iFrame<size; ++iFrame) {
      if (!packetisedData_) {
	data.push_back( txFileReader_.get(iAmc).link(i).at(iFrame) );
      }
    }
    
    Block block(id, data);

    blocks_.push_back(block);

  }

  // advance pointers to next event
  if (!packetisedData_) {
    rxIndex_ += nFramesPerEvent_;
    txIndex_ += nFramesPerEvent_;
  }
  
}    
  
  
void
MP7BufferDumpToRaw::formatRaw(FEDRawData& feddata, int bx, int evtId, int orbit)
{

  // now create the raw data array
  int capEvtSize = 0;
  for (std::vector<Block>::const_iterator itr=blocks_.begin(); itr!=blocks_.end(); ++itr) {
    capEvtSize += itr->getSize() + 1;
  }

  int amcSize = 0;
  for (unsigned i=0; i<rxBlockLength_.size(); ++i) amcSize += 4 * (rxBlockLength_.at(i) + 1);
  for (unsigned i=0; i<txBlockLength_.size(); ++i) amcSize += 4 * (txBlockLength_.at(i) + 1);

  int fedSize = amcSize;
  fedSize += lenSlinkHeader_;
  fedSize += lenSlinkTrailer_;
  fedSize += lenAMC13Header_;
  fedSize += lenAMC13Trailer_;
  fedSize += lenAMCHeader_;
  fedSize += lenAMCTrailer_;
  fedSize = (fedSize+7) & ~0x7;// round up to multiple of 8

  edm::LogInfo("mp7") << "Captured event size=" << capEvtSize << ", AMCsize=" << amcSize << ", FED size=" << fedSize << std::endl;

  // event info for headers
  edm::LogInfo("mp7") << "Event : " << evtId << " orbit=" << orbit << " bx=" << bx << std::endl;


  // Allocate space for header+trailer+payload
  feddata.resize(fedSize);

  int iWord = 0;

  // write SLINK header
  feddata.data()[iWord+1] |= fedId_ & 0xff;
  feddata.data()[iWord+2] |= (fedId_>>8) & 0xf;

  feddata.data()[iWord+2] |= (bx<<4) & 0xff;
  feddata.data()[iWord+3] |= (bx>>4) & 0xff;

  feddata.data()[iWord+4] = evtId & 0xff;
  feddata.data()[iWord+5] = (evtId>>8) & 0xff;
  feddata.data()[iWord+6] = (evtId>>16) & 0xff;

  feddata.data()[iWord+7] |= evType_ & 0xf;
  feddata.data()[iWord+7] |= 0x50;


  // write AMC13 header
  iWord += lenSlinkHeader_;

  // do nothing for now

  // write AMC header
  iWord += lenAMC13Header_;

  feddata.data()[iWord+0] = evtId & 0xff;
  feddata.data()[iWord+1] = (evtId>>8) & 0xff;
  feddata.data()[iWord+2] = (evtId>>16) & 0xff;

  feddata.data()[iWord+4] |= (orbit<<4) & 0xff;
  feddata.data()[iWord+5] |= (orbit>>4) & 0xff;
  feddata.data()[iWord+6] |= (orbit>>12) & 0xff;

  feddata.data()[iWord+6] |= (bx>>4) & 0xff;
  feddata.data()[iWord+7] |= (bx>>12) & 0xff;

  feddata.data()[iWord+8] = evType_ & 0xff;

  feddata.data()[iWord+9]  = (amcSize/4) & 0xff;
  feddata.data()[iWord+10] = ((amcSize/4) >> 8) & 0xff;

  feddata.data()[iWord+11] = fwVer_ & 0xff;


  // now add payload

  iWord += lenAMCHeader_;
  std::ostringstream payloadInfo;

  //  for (int iBlock=0; iBlock<nRxLinks_ && iWord<fedSize; ++iBlock) {

  //    int blockId     = 2*iBlock;
  //    int blockLength = rxBlockLength_.at(iBlock);

    // write block header
  //    feddata.data()[iWord+2] = blockLength & 0xff;
  //    feddata.data()[iWord+3] = blockId & 0xff;
  //    iWord+=4;

  //    if (blockLength!=0) {
  //      payloadInfo << ", " << (blockId&0xff) << " (" << (blockLength&0xff) << ")";
  //    }

  //    if (blockLength>(int)rxData.at(iBlock).size()) {
  //      edm::LogError("mp7") << "Read insufficient data for block " << blockId <<". Expected " << rxBlockLength_.at(iBlock) << " read " << rxData.at(iBlock).size() << " from Rx link " << iBlock << std::endl;
  //      continue;
  //    }

  //    for (int i=0; i<blockLength; ++i) {
  //      if(i < (int)rxData.at(iBlock).size()) {
  //	feddata.data()[iWord]   = rxData.at(iBlock).at(i) & 0xff;
  //	feddata.data()[iWord+1] = (rxData.at(iBlock).at(i) >> 8) & 0xff;
  //	feddata.data()[iWord+2] = (rxData.at(iBlock).at(i) >> 16) & 0xff;
  //	feddata.data()[iWord+3] = (rxData.at(iBlock).at(i) >> 24) & 0xff;
  //	iWord+=4;
  //      }
  //      else {
  //	feddata.data()[iWord]   = 0;
  //	feddata.data()[iWord+1] = 0;
  //	feddata.data()[iWord+2] = 0;
  //	feddata.data()[iWord+3] = 0;
  //	iWord+=4;
  //      }
  //    }

  //  }

  //  edm::LogInfo("mp7") << "Rx blocks : " << payloadInfo.str() << std::endl;

  // now do Tx links
  // strictly these will appear in the wrong place
  // they should be interspersed with Rx channels, not appended

  //  payloadInfo.str("");

  //  for (int iBlock=0; iBlock<nTxLinks_ && iWord<fedSize; ++iBlock) {

  //    int blockId     = (2*iBlock)+1;
  //    int blockLength = txBlockLength_.at(iBlock);

  //    // write block header
  //    feddata.data()[iWord+2] = blockLength & 0xff;
  //    feddata.data()[iWord+3] = blockId & 0xff;
  //    iWord+=4;

  //    payloadInfo << ", " << (blockId&0xff) << "(" << (blockLength&0xff) << ")";

  //    if (blockLength>(int)txData.at(iBlock).size()) {
  //      edm::LogError("mp7") << "Read insufficient data for block " << blockId <<". Expected " << blockLength << " read " << txData.at(iBlock).size() << " from Tx link " << iBlock << std::endl;
  //      continue;
  //    }

  //    for (int i=0; i<blockLength; ++i) {
  //      if (i<(int)txData.at(iBlock).size()) {
  //	feddata.data()[iWord]   = txData.at(iBlock).at(i) & 0xff;
  //	feddata.data()[iWord+1] = (txData.at(iBlock).at(i) >> 8) & 0xff;
  //	feddata.data()[iWord+2] = (txData.at(iBlock).at(i) >> 16) & 0xff;
  //	feddata.data()[iWord+3] = (txData.at(iBlock).at(i) >> 24) & 0xff;
  //	iWord+=4;
  //      }
  //      else {
  //	feddata.data()[iWord]   = 0;
  //	feddata.data()[iWord+1] = 0;
  //	feddata.data()[iWord+2] = 0;
  //	feddata.data()[iWord+3] = 0;
  //	iWord+=4;
  //      }
  //    }

  //  }

  //  edm::LogInfo("mp7") << "Tx blocks : " << payloadInfo.str() << std::endl;

  // write AMC trailer
  feddata.data()[iWord]   = evtId & 0xff;
  feddata.data()[iWord+1] = (evtId>>8) & 0xff;

  // write AMC13 trailer
  iWord += lenAMCTrailer_;

  // write SLINK trailer
  iWord += lenAMC13Trailer_;
  iWord = (iWord+7) & ~0x7;  // move to next 64 bit boundary
  feddata.data()[iWord+4] = (fedSize/8) & 0xff;
  feddata.data()[iWord+5] = ((fedSize/8)>>8) & 0xff;
  feddata.data()[iWord+6] = ((fedSize/8)>>16) & 0xff;

  feddata.data()[iWord+7] |= 0xa0;

  edm::LogInfo("mp7") << "End of packet after " << iWord+8 << " bytes  of " << fedSize << std::endl;


}


// ------------ method called once each job just before starting event loop  ------------
void 
MP7BufferDumpToRaw::beginJob()
{



}


// ------------ method called once each job just after ending the event loop  ------------
void 
MP7BufferDumpToRaw::endJob() 
{


}

// ------------ method called when starting to processes a run  ------------
/*
void 
MP7BufferDumpToRaw::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
MP7BufferDumpToRaw::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
vvoid 
MP7BufferDumpToRaw::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
MP7BufferDumpToRaw::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MP7BufferDumpToRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

}

using namespace l1t;
//define this as a plug-in
DEFINE_FWK_MODULE(MP7BufferDumpToRaw);
