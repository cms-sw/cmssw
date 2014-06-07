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

  // constants
  int fedId_;
  int nHeaders_;
  int txBlockOffset_;
  int nFramesPerEvent_;
  int nRxLinks_;
  int nTxLinks_;

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

  fedId_ = iConfig.getUntrackedParameter<int>("fedId", 1);

  nHeaders_ = iConfig.getUntrackedParameter<int>("nHeaders", 3);
  nFramesPerEvent_ = iConfig.getUntrackedParameter<int>("nFramesPerEvent", 32);
  txBlockOffset_ = iConfig.getUntrackedParameter<int>("txBlockOffset", 72);

  nRxLinks_ = iConfig.getUntrackedParameter<int>("nRxLinks", 72);
  nTxLinks_ = iConfig.getUntrackedParameter<int>("nTxLinks", 72);

  edm::LogInfo("1t|mp7") << "nRxLinks = " << nRxLinks_ << " nTxLinks=" << nTxLinks_ << std::endl;

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
    std::vector<std::string> data;
    boost::split(data, line, boost::is_any_of("\t "));
    std::vector<std::string>::iterator tmp = data.begin();
    tmp++; tmp++; tmp++;
    data.erase(data.begin(), tmp);
    
    // check we have read the right number of link words
    if ((int)data.size() != nRxLinks_) {
      edm::LogError("l1t|mp7") << "Read " << data.size() << " Rx links, expected " << nRxLinks_ << std::endl;
    }

    for (int j=0; j<nRxLinks_ && j<(int)data.size(); ++j) {
      if (i<(int)rxData.size() && j<(int)rxData.at(i).size() && j<(int)data.size()) {
	data.at(j).erase(0,2);  // remove 1v from start of word
	rxData.at(i).at(j) = std::stoul(data.at(j), nullptr, 16);
      }
      else edm::LogError("l1t|mp7") << "Error reading Rx file. i=" << i << " j=" << j << " rxData.size()=" << rxData.size() << " data.size()=" << data.size() << std::endl;
    }
    
    // output buffers
    std::getline(txFile_, line);
    boost::split(data, line, boost::is_any_of("\t "));
    tmp = data.begin();
    tmp++; tmp++; tmp++;
    data.erase(data.begin(), tmp);

    // check we have read the right number of link words
    if ((int)data.size() != nTxLinks_) {
      edm::LogError("l1t|mp7") << "Read " << data.size() << " Tx links, expected " << nTxLinks_ << std::endl;
    }

    for (int j=0; j<nTxLinks_ && j<(int)data.size(); ++j) {
      if (i<(int)txData.size() && j<(int)txData.at(i).size() && j<(int)data.size()) {
	data.at(j).erase(0,2);  // remove 1v from start of word
	txData.at(i).at(j) = std::stoul(data.at(j), nullptr, 16);
      }
      else edm::LogError("l1t|mp7") << "Error reading Tx file. i=" << i << " j=" << j << " txData.size()=" << txData.size() << " data.size()=" << data.size() << std::endl;
    }
  
  }

  // check size of vectors now !

  // now create the raw data array
  int nBlocks  = nRxLinks_ + nTxLinks_;
  int blockSize = nFramesPerEvent_+1; 
  int evtSize  = nBlocks * blockSize;

  // create the collection
  std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 
  
  // retrieve the target buffer
  FEDRawData& feddata=rawColl->FEDData(fedId_);
  
  // Allocate space for header+trailer+payload
  feddata.resize(evtSize);
  int iWord=0;

  for (int iBlock=0; iBlock<nRxLinks_ && iWord<evtSize; ++iBlock) {

    feddata.data()[iWord+2] = nFramesPerEvent_ & 0xf;
    feddata.data()[iWord+3] = iBlock & 0xf ;
    iWord+=4;

    for (int i=0; i<nFramesPerEvent_; ++i) {
      if (i<(int)rxData.size() && iBlock<(int)rxData.at(i).size()) {
	feddata.data()[iWord] = rxData.at(i).at(iBlock);
	iWord+=4;
      }
      else edm::LogError("l1t|mp7") << "Error. i=" << i << " iBlock=" << iBlock << std::endl;
    }

  }

  for (int iBlock=0; iBlock<nTxLinks_ && iWord<evtSize; ++iBlock) {

    feddata.data()[iWord+2] = nFramesPerEvent_ & 0xf;
    feddata.data()[iWord+3] = (iBlock+txBlockOffset_) & 0xf;
    iWord+=4;

    for (int i=0; i<nFramesPerEvent_; ++i) {
      if (i<(int)txData.size() && iBlock<(int)txData.at(i).size()) {
	feddata.data()[iWord] = txData.at(i).at(iBlock);
	iWord+=4;
      }
      else edm::LogError("l1t|mp7") << "Error. i=" << i << " iBlock=" << iBlock << std::endl;
    }

  }

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
  for (int i=0; i<nHeaders_; ++i) {
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
