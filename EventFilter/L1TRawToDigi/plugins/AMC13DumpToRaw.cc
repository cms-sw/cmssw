// -*- C++ -*-
//
// Package:    EventFilter/L1TRawToDigi
// Class:      AMC13DumpToRaw
// 
/**\class AMC13DumpToRaw AMC13DumpToRaw.cc L1Trigger/L1TCalorimeter/plugins/AMC13DumpToRaw.cc

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
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "FWCore/Utilities/interface/CRC16.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <boost/algorithm/string.hpp>

#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"


namespace l1t {

class AMC13DumpToRaw : public edm::EDProducer {
public:
  explicit AMC13DumpToRaw(const edm::ParameterSet&);
  ~AMC13DumpToRaw();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  void readEvent(std::vector<uint32_t>& load32);

  //  void formatAMC(amc13::Packet& amc13, const std::vector<uint32_t>& load32);

  //  void formatRaw(edm::Event& iEvent, amc13::Packet& amc13, FEDRawData& fed_data);
  
  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------
  std::ifstream file_;
  std::string filename_;

  // DAQ params
  int fedId_;
  int iAmc_;
  int boardId_;
  int evType_;
  int fwVer_;
  int slinkHeaderSize_;  // in 8-bit words
  int slinkTrailerSize_;

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
  AMC13DumpToRaw::AMC13DumpToRaw(const edm::ParameterSet& iConfig) :
    filename_(iConfig.getUntrackedParameter<std::string>("filename", "data.txt")),
    fedId_(iConfig.getUntrackedParameter<int>("fedId", 1)),
    iAmc_(iConfig.getUntrackedParameter<int>("iAmc", 1)),
    boardId_(iConfig.getUntrackedParameter<int>("boardId", 1)),
    evType_(iConfig.getUntrackedParameter<int>("eventType", 1)),
    fwVer_(iConfig.getUntrackedParameter<int>("fwVersion", 1)),
    slinkHeaderSize_(iConfig.getUntrackedParameter<int>("lenSlinkHeader", 8)),
    slinkTrailerSize_(iConfig.getUntrackedParameter<int>("lenSlinkTrailer", 8))
{

  produces<FEDRawDataCollection>();

}


AMC13DumpToRaw::~AMC13DumpToRaw()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
AMC13DumpToRaw::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // create AMC 13 packet
  // amc13::Packet amc13;    

  std::vector<uint32_t> load32;

  readEvent(load32);
  
  //  formatAMC(amc13, load32);

  int size = load32.size() * 4;

  LogDebug("L1T") << "AMC13 size " << size << " bytes";  

  // prepare the raw data collection
  std::auto_ptr<FEDRawDataCollection> raw_coll(new FEDRawDataCollection());
  FEDRawData& fed_data = raw_coll->FEDData(fedId_);

  fed_data.resize(size);

  // fill FEDRawData object
  for (unsigned i=0; i<load32.size(); ++i) {
    for (unsigned j=0; j<4; ++j) {
      fed_data.data()[i*4+j] = (load32.at(i)>>(8*j)) & 0xff;
    }
  }

  //  formatRaw(iEvent, amc13, fed_data);

  LogDebug("L1T") << "Packing FED ID " << fedId_ << " size " << fed_data.size();
  
  // put the collection in the event
  iEvent.put(raw_coll);  

}


void
AMC13DumpToRaw::readEvent(std::vector<uint32_t>& load32) {

  // read file
  std::string line;
  
  // while not encountering dumb errors
  while (getline(file_, line) && !line.empty() ) {
    
    std::istringstream iss(line);
    unsigned long d;
    iss >> std::hex >> d;
    
    load32.push_back( d ) ;

  }

}


// void
// AMC13DumpToRaw::formatAMC(amc13::Packet& amc13, const std::vector<uint32_t>& load32) {

//   // TODO this is an empty word to be replaced with a proper MP7
//   // header containing at least the firmware version
  
//   std::vector<uint64_t> load64;
//   for (unsigned int i = 0; i < load32.size(); i += 2) {
//     uint64_t word = load32[i];
//     if (i + 1 < load32.size())
//       word |= static_cast<uint64_t>(load32[i + 1]) << 32;
//     load64.push_back(word);
//   }
  
//   LogDebug("L1T") << "Creating AMC packet " << iAmc_;
  
//   amc13.add(iAmc_, boardId_, load64);

// }

  
  
// void
// AMC13DumpToRaw::formatRaw(edm::Event& iEvent, amc13::Packet& amc13, FEDRawData& fed_data)
// {

//   unsigned int size = slinkHeaderSize_ + slinkTrailerSize_ + amc13.size() * 8;
//   fed_data.resize(size);
//   unsigned char * payload = fed_data.data();
//   unsigned char * payload_start = payload;

//   auto bxId = iEvent.bunchCrossing();
//   auto evtId = iEvent.id().event();

//   LogDebug("L1T") << "Creating FEDRawData ID " << fedId_ << ", size " << size;

//   FEDHeader header(payload);
//   header.set(payload, evType_, evtId, bxId, fedId_);

//   payload += slinkHeaderSize_;

//   amc13.write(iEvent, payload, size - slinkHeaderSize_ - slinkTrailerSize_);

//   payload += amc13.size() * 8;

//   FEDTrailer trailer(payload);
//   trailer.set(payload, size / 8, evf::compute_crc(payload_start, size), 0, 0);

// }


// ------------ method called once each job just before starting event loop  ------------
void 
AMC13DumpToRaw::beginJob()
{

  // open VME file
  file_.open(filename_.c_str(), std::ios::in);
  if(!file_.good()) { edm::LogInfo("TextToDigi") << "Failed to open ASCII file " << filename_ << std::endl; }


}


// ------------ method called once each job just after ending the event loop  ------------
void 
AMC13DumpToRaw::endJob() 
{
  
  file_.close();

}

// ------------ method called when starting to processes a run  ------------
/*
void 
AMC13DumpToRaw::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
AMC13DumpToRaw::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
vvoid 
AMC13DumpToRaw::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
AMC13DumpToRaw::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
AMC13DumpToRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

}

using namespace l1t;
//define this as a plug-in
DEFINE_FWK_MODULE(AMC13DumpToRaw);
