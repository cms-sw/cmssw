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
  //  int txBlockOffset_;
  int nFramesPerEvent_;
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
  nFramesPerEvent_ = iConfig.getUntrackedParameter<int>("nFramesPerEvent", 32);
  //  txBlockOffset_ = iConfig.getUntrackedParameter<int>("txBlockOffset", 72);

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
  
  // array of data (frame, link)
  std::vector< std::vector< int > > rxData;
  std::vector< std::vector< int > > txData;

  rxData.resize(nFramesPerEvent_, std::vector<int>(nRxLinks_, 0) );
  txData.resize(nFramesPerEvent_, std::vector<int>(nTxLinks_, 0) );
  

  // read lines from file
  for (int i=0; i<nFramesPerEvent_; ++i) {

    std::string line;
    
    // input buffers
    std::getline(rxFile_, line);
    if (!rxFile_) {
      edm::LogError("mp7") << "End of Rx input file!" << std::endl;
    }

    std::vector<std::string> data;
    boost::split(data, line, boost::is_any_of("\t "));
    std::vector<std::string>::iterator tmp = data.begin();
    tmp++; tmp++; tmp++;
    data.erase(data.begin(), tmp);
    
    // check we have read the right number of link words
    if ((int)data.size() != nRxLinks_) {
      edm::LogError("mp7") << "Read " << data.size() << " Rx links, expected " << nRxLinks_ << std::endl;
    }

    for (int j=0; j<nRxLinks_ && j<(int)data.size(); ++j) {
      if (i<(int)rxData.size() && j<(int)rxData.at(i).size() && j<(int)data.size()) {
	data.at(j).erase(0,2);  // remove 1v from start of word
	rxData.at(i).at(j) = std::stoul(data.at(j), nullptr, 16);
      }
      else edm::LogError("mp7") << "Error reading Rx file. i=" << i << " j=" << j << " rxData.size()=" << rxData.size() << " data.size()=" << data.size() << std::endl;
    }
    
    // output buffers
    std::getline(txFile_, line);
    if (!txFile_) {
      edm::LogError("mp7") << "End of Tx input file!" << std::endl;
    }

    boost::split(data, line, boost::is_any_of("\t "));
    tmp = data.begin();
    tmp++; tmp++; tmp++;
    data.erase(data.begin(), tmp);

    // check we have read the right number of link words
    if ((int)data.size() != nTxLinks_) {
      edm::LogError("mp7") << "Read " << data.size() << " Tx links, expected " << nTxLinks_ << std::endl;
    }

    for (int j=0; j<nTxLinks_ && j<(int)data.size(); ++j) {
      if (i<(int)txData.size() && j<(int)txData.at(i).size() && j<(int)data.size()) {
	data.at(j).erase(0,2);  // remove 1v from start of word
	txData.at(i).at(j) = std::stoul(data.at(j), nullptr, 16);
      }
      else edm::LogError("mp7") << "Error reading Tx file. i=" << i << " j=" << j << " txData.size()=" << txData.size() << " data.size()=" << data.size() << std::endl;
    }
  
  }

  // check size of vectors now !

  // now create the raw data array
  int capEvtSize = (nRxLinks_ + nTxLinks_) * (nFramesPerEvent_+1)*4;

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

  for (int iBlock=0; iBlock<nRxLinks_ && iWord<fedSize; ++iBlock) {

    int blockId     = 2*iBlock;
    int blockLength = rxBlockLength_.at(iBlock);

    // write block header
    feddata.data()[iWord+2] = blockLength & 0xff;
    feddata.data()[iWord+3] = blockId & 0xff;
    iWord+=4;

    edm::LogInfo("mp7") << "Rx Block : ID=" << (blockId&0xff) << " Length=" << (blockLength&0xff) << std::endl;

    for (int i=0; i<blockLength; ++i) {
      if (i<(int)rxData.size() && iBlock<(int)rxData.at(i).size()) {
	feddata.data()[iWord]   = rxData.at(i).at(iBlock) & 0xff;
	feddata.data()[iWord+1] = (rxData.at(i).at(iBlock) >> 8) & 0xff;
	feddata.data()[iWord+2] = (rxData.at(i).at(iBlock) >> 16) & 0xff;
	feddata.data()[iWord+3] = (rxData.at(i).at(iBlock) >> 24) & 0xff;
	iWord+=4;
      }
      else edm::LogError("mp7") << "Error. i=" << i << " iBlock=" << iBlock << std::endl;
    }

  }

  // now do Tx links
  // strictly these will appear in the wrong place
  // they should be interspersed with Rx channels, not appended

  for (int iBlock=0; iBlock<nTxLinks_ && iWord<fedSize; ++iBlock) {

    int blockId     = (2*iBlock)+1;
    int blockLength = txBlockLength_.at(iBlock);

    // write block header
    feddata.data()[iWord+2] = blockLength & 0xff;
    feddata.data()[iWord+3] = blockId & 0xff;
    iWord+=4;

    edm::LogInfo("mp7") << "Tx Block : ID=" << (blockId&0xff) << " Length=" << (blockLength&0xff) << std::endl;

    for (int i=0; i<blockLength; ++i) {
      if (i<(int)txData.size() && iBlock<(int)txData.at(i).size()) {
	feddata.data()[iWord]   = txData.at(i).at(iBlock) & 0xff;
	feddata.data()[iWord+1] = (txData.at(i).at(iBlock) >> 8) & 0xff;
	feddata.data()[iWord+2] = (txData.at(i).at(iBlock) >> 16) & 0xff;
	feddata.data()[iWord+3] = (txData.at(i).at(iBlock) >> 24) & 0xff;
	iWord+=4;
      }
      else edm::LogError("mp7") << "Error. i=" << i << " iBlock=" << iBlock << std::endl;
    }

  }

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

}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::MP7BufferDumpToRaw);
