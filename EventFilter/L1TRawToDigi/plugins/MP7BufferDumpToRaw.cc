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

  void findNextEvent();
  std::vector<std::string> readLine(std::ifstream& file, int nLinks);
  
  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------

  // input files
  std::string rxFilename_;
  std::string txFilename_;
  std::ifstream rxFile_;
  std::ifstream txFile_;

  // input file parameters
  int nTextHeaderLines_;
  int nFramesPerEvent_;
  int txLatency_;
  int nRxLinks_;
  int nTxLinks_;

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
MP7BufferDumpToRaw::MP7BufferDumpToRaw(const edm::ParameterSet& iConfig)
{

  produces<FEDRawDataCollection>();

  //now do what ever initialization is needed
  rxFilename_ = iConfig.getUntrackedParameter<std::string>("rxFile", "rx_summary.txt");
  txFilename_ = iConfig.getUntrackedParameter<std::string>("txFile", "tx_summary.txt");

  nTextHeaderLines_ = iConfig.getUntrackedParameter<int>("nTextHeaderLines", 3);
  nFramesPerEvent_ = iConfig.getUntrackedParameter<int>("nFramesPerEvent", 41);
  txLatency_= iConfig.getUntrackedParameter<int>("txLatency", 61);

  nRxLinks_ = iConfig.getUntrackedParameter<int>("nRxLinks", 72);
  nTxLinks_ = iConfig.getUntrackedParameter<int>("nTxLinks", 72);


  // DAQ parameters
  fedId_ = iConfig.getUntrackedParameter<int>("fedId", 1);
  evType_ = iConfig.getUntrackedParameter<int>("eventType", 1);
  fwVer_ = iConfig.getUntrackedParameter<int>("fwVersion", 1);
  lenSlinkHeader_ = iConfig.getUntrackedParameter<int>("lenSlinkHeader", 16);
  lenSlinkTrailer_ = iConfig.getUntrackedParameter<int>("lenSlinkTrailer", 16);
  lenAMC13Header_ = iConfig.getUntrackedParameter<int>("lenAMC13Header", 0);
  lenAMC13Trailer_ = iConfig.getUntrackedParameter<int>("lenAMC13Trailer", 0);
  lenAMCHeader_   = iConfig.getUntrackedParameter<int>("lenAMCHeader", 12);
  lenAMCTrailer_   = iConfig.getUntrackedParameter<int>("lenAMCTrailer", 8);

  rxBlockLength_ = iConfig.getUntrackedParameter< std::vector<int> >("rxBlockLength");
  txBlockLength_ = iConfig.getUntrackedParameter< std::vector<int> >("txBlockLength");

  if (rxBlockLength_.size() != (unsigned) nRxLinks_) {
    edm::LogError("mp7") << "Inconsistent configuration : N block lengths=" << rxBlockLength_.size() << " for " << nRxLinks_ << " Rx links" << std::endl;
  }

  if (txBlockLength_.size() != (unsigned) nTxLinks_) {
    edm::LogError("mp7") << "Inconsistent configuration : N block lengths=" << txBlockLength_.size() << " for " << nTxLinks_ << " Tx links" << std::endl;
  }


  edm::LogInfo("mp7") << "nRxLinks = " << nRxLinks_ << " nTxLinks=" << nTxLinks_ << std::endl;

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

  // read forward to next valid event
  findNextEvent();
  
  // array of data (frame, link)
  std::vector< std::vector< int > > rxData(nRxLinks_, std::vector<int>(0));
  std::vector< std::vector< int > > txData(nTxLinks_, std::vector<int>(0));

  // read lines from file
  int nFramesRead = 0;
  while (nFramesRead < nFramesPerEvent_) {
    
    std::vector<std::string> rxStrData = readLine(rxFile_, nRxLinks_);
    std::vector<std::string> txStrData = readLine(txFile_, nTxLinks_);

    // store data
    int nRxValidLinks(0), nTxValidLinks(0);
    for (int iLink=0; iLink<nRxLinks_; ++iLink) {

      // check data is valid
      int dataValid = std::stoul(rxStrData.at(iLink+1).substr(0,1));
      if (dataValid==1) {
	rxStrData.at(iLink+1).erase(0,2);  // remove 1v from start of word
	rxData.at(iLink).push_back( std::stoul(rxStrData.at(iLink+1), nullptr, 16) );
	nRxValidLinks++;
      }

    }

    // store data
    for (int iLink=0; iLink<nTxLinks_; ++iLink) {
      // check data is valid
      int dataValid = std::stoul(txStrData.at(iLink+1).substr(0,1));
      if (dataValid==1) {
	txStrData.at(iLink+1).erase(0,2);  // remove 1v from start of word
	txData.at(iLink).push_back( std::stoul(txStrData.at(iLink+1), nullptr, 16) );
	nTxValidLinks++;
      }

    }

    LogDebug("mp7") << "Rx Frame " << rxStrData.at(0) << " " << nRxValidLinks << ", Tx Frame " << txStrData.at(0) << " " << nTxValidLinks << std::endl;
    
    nFramesRead++;

  }

  // check size of vectors
  int maxRxFrames=0;
  int totalRxWord32s=0;
  std::ostringstream rxInfo;
  rxInfo << "Rx data : ";
  for (int iLink=0; iLink<nRxLinks_; iLink++) {
    int nf = (int) rxData.at(iLink).size();
    if (nf > maxRxFrames) maxRxFrames = nf;
    totalRxWord32s+= nf;
    rxInfo << nf << " ";
  }
  edm::LogInfo("mp7") << rxInfo.str() << std::endl;

  int maxTxFrames=0;
  int totalTxWord32s=0;
  std::ostringstream txInfo;
  txInfo << "Tx data : ";
  for (int iLink=0; iLink<nTxLinks_; iLink++) {
    int nf = (int) txData.at(iLink).size();
    if (nf > maxTxFrames) maxTxFrames = nf;
    totalTxWord32s += nf;
    txInfo << nf << " ";
  }
  edm::LogInfo("mp7") << txInfo.str() << std::endl;
  
  edm::LogInfo("mp7") << "Rx summary : Max frames=" << maxRxFrames << " total word32s=" << totalRxWord32s << std::endl;
  edm::LogInfo("mp7") << "Tx summary : Max frames=" << maxTxFrames << " total word32s=" << totalTxWord32s << std::endl;

  // captured data size
  


  if (maxRxFrames==0 || maxTxFrames==0) return;

  // now create the raw data array
  int capEvtSize = (totalRxWord32s+totalTxWord32s)*4;

  int amcSize = 0;
  for (int i=0; i<nRxLinks_; ++i) amcSize += 4 * (rxBlockLength_.at(i) + 1);
  for (int i=0; i<nTxLinks_; ++i) amcSize += 4 * (txBlockLength_.at(i) + 1);

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
  int bx = iEvent.bunchCrossing();
  int evtId = iEvent.id().event();
  long int orbit = iEvent.orbitNumber();
  
  edm::LogInfo("mp7") << "Event : " << evtId << " orbit=" << orbit << " bx=" << bx << std::endl;


  // create the collection
  std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 
  
  // retrieve the target buffer
  FEDRawData& feddata=rawColl->FEDData(fedId_);
  
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

  for (int iBlock=0; iBlock<nRxLinks_ && iWord<fedSize; ++iBlock) {

    int blockId     = 2*iBlock;
    int blockLength = rxBlockLength_.at(iBlock);

    // write block header
    feddata.data()[iWord+2] = blockLength & 0xff;
    feddata.data()[iWord+3] = blockId & 0xff;
    iWord+=4;

    if (blockLength!=0) {
      payloadInfo << ", " << (blockId&0xff) << " (" << (blockLength&0xff) << ")";
    }

    if (blockLength>(int)rxData.at(iBlock).size()) {
      edm::LogError("mp7") << "Read insufficient data for block " << blockId <<". Expected " << rxBlockLength_.at(iBlock) << " read " << rxData.at(iBlock).size() << " from Rx link " << iBlock << std::endl;
      continue;
    }

    for (int i=0; i<blockLength; ++i) {
      if(i < (int)rxData.at(iBlock).size()) {
	feddata.data()[iWord]   = rxData.at(iBlock).at(i) & 0xff;
	feddata.data()[iWord+1] = (rxData.at(iBlock).at(i) >> 8) & 0xff;
	feddata.data()[iWord+2] = (rxData.at(iBlock).at(i) >> 16) & 0xff;
	feddata.data()[iWord+3] = (rxData.at(iBlock).at(i) >> 24) & 0xff;
	iWord+=4;
      }
      else {
	feddata.data()[iWord]   = 0;
	feddata.data()[iWord+1] = 0;
	feddata.data()[iWord+2] = 0;
	feddata.data()[iWord+3] = 0;
	iWord+=4;
      }
    }

  }

  edm::LogInfo("mp7") << "Rx blocks : " << payloadInfo.str() << std::endl;

  // now do Tx links
  // strictly these will appear in the wrong place
  // they should be interspersed with Rx channels, not appended

  payloadInfo.str("");

  for (int iBlock=0; iBlock<nTxLinks_ && iWord<fedSize; ++iBlock) {

    int blockId     = (2*iBlock)+1;
    int blockLength = txBlockLength_.at(iBlock);

    // write block header
    feddata.data()[iWord+2] = blockLength & 0xff;
    feddata.data()[iWord+3] = blockId & 0xff;
    iWord+=4;

    payloadInfo << ", " << (blockId&0xff) << "(" << (blockLength&0xff) << ")";

    if (blockLength>(int)txData.at(iBlock).size()) {
      edm::LogError("mp7") << "Read insufficient data for block " << blockId <<". Expected " << blockLength << " read " << txData.at(iBlock).size() << " from Tx link " << iBlock << std::endl;
      continue;
    }

    for (int i=0; i<blockLength; ++i) {
      if (i<(int)txData.at(iBlock).size()) {
	feddata.data()[iWord]   = txData.at(iBlock).at(i) & 0xff;
	feddata.data()[iWord+1] = (txData.at(iBlock).at(i) >> 8) & 0xff;
	feddata.data()[iWord+2] = (txData.at(iBlock).at(i) >> 16) & 0xff;
	feddata.data()[iWord+3] = (txData.at(iBlock).at(i) >> 24) & 0xff;
	iWord+=4;
      }
      else {
	feddata.data()[iWord]   = 0;
	feddata.data()[iWord+1] = 0;
	feddata.data()[iWord+2] = 0;
	feddata.data()[iWord+3] = 0;
	iWord+=4;
      }
    }

  }

  edm::LogInfo("mp7") << "Tx blocks : " << payloadInfo.str() << std::endl;

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

  // put the collection in the event
  iEvent.put(rawColl);

}


// ------------ method called once each job just before starting event loop  ------------
void 
MP7BufferDumpToRaw::beginJob()
{

  // open files and read headers
  rxFile_.open(rxFilename_.c_str());
  txFile_.open(txFilename_.c_str());

  std::string line;
  for (int i=0; i<nTextHeaderLines_; ++i) {
    std::getline(rxFile_, line);
  }

  for (int i=0; i<nTextHeaderLines_+txLatency_-1; ++i) {
    std::getline(txFile_, line);
  }

}


// ------------ method called once each job just after ending the event loop  ------------
void 
MP7BufferDumpToRaw::endJob() 
{

  // close files
  rxFile_.close();
  txFile_.close();

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
void 
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

void
MP7BufferDumpToRaw::findNextEvent() {

  // find the first event
  int iFrame = 0 ;
  int lastFlag=1;
  bool dataValid = false;
  while (!dataValid) {

    std::vector<std::string> rxData = readLine(rxFile_, nRxLinks_);
    std::vector<std::string> txData = readLine(txFile_, nTxLinks_);

    iFrame = std::stoul(rxData.at(0));

    // check for data valid in first link
    int newFlag = std::stoul(rxData.at(1).substr(0,1));
    if (newFlag==1 && lastFlag==0) {
      dataValid = true;
      edm::LogInfo("mp7") << "Found Rx event at frame " << iFrame << std::endl;
    }
    else {
      lastFlag = newFlag;
    }

  }

  // check for data going high
  dataValid = false;
  while (!dataValid) {

    std::vector<std::string> txData = readLine(txFile_, nTxLinks_);

    iFrame = std::stoul(txData.at(0));

    // check for data valid in first link
    int newFlag = std::stoul(txData.at(1).substr(0,1));
    if (newFlag==1 && lastFlag==0) {
      dataValid = true;
      edm::LogInfo("mp7") << "Found Tx event at frame " << iFrame << std::endl;
    }
    else {
      lastFlag = newFlag;
    }

  }
  


}

std::vector<std::string>
MP7BufferDumpToRaw::readLine(std::ifstream& file, int nLinks) {

  // input buffers
  std::string line;
  std::getline(file, line);
  if (!file) {
    edm::LogError("mp7") << "End of input file! " << file << std::endl;
    return std::vector<std::string>(0);
  }
  
  // split line into tokens
  std::vector<std::string> data;
  boost::split(data, line, boost::is_any_of("\t "),boost::token_compress_on);
  
  // check we have read the right number of link words
  if ((int)data.size()-3 != nLinks) {
    edm::LogError("mp7") << "Read " << data.size() << " links, expected " << nLinks << " " << data.at(0) << " " << data.at(data.size()-1) << std::endl;
    return std::vector<std::string>(0);
  }
  
  // remove "Frame" and ":"
  std::vector<std::string>::iterator itr = data.begin();
  data.erase(itr);
  itr++;
  data.erase(itr);

  return data;
  
}

}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::MP7BufferDumpToRaw);
