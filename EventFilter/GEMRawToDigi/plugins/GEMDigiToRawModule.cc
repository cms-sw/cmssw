/** \file
 *  \author J. Lee - UoS
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/GEMRawToDigi/plugins/GEMDigiToRawModule.h"

using namespace gem;

GEMDigiToRawModule::GEMDigiToRawModule(const edm::ParameterSet & pset) 
{
  event_type_ = pset.getParameter<int>("eventType");
  digi_token = consumes<GEMDigiCollection>( pset.getParameter<edm::InputTag>("gemDigi") );
  produces<FEDRawDataCollection>("GEMRawData");
}

void GEMDigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gemDigi", edm::InputTag("simMuonGEMDigis"));
}

void GEMDigiToRawModule::beginRun(const edm::Run &run, const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMEMap> gemEMap;
  iSetup.get<GEMEMapRcd>().get(gemEMap); 
  m_gemEMap = gemEMap.product();
  m_gemROMap = m_gemEMap->convert();

}

void GEMDigiToRawModule::produce( edm::Event & e, const edm::EventSetup& c )
{
  bool verbose_ = true;

  auto fedRawDataCol = std::make_unique<FEDRawDataCollection>();

  // Take digis from the event
  edm::Handle<GEMDigiCollection> gemDigis;
  e.getByToken( digi_token, gemDigis );

  std::vector<AMC13Event*> amc13Events;
    
  for (GEMDigiCollection::DigiRangeIterator gemdgIt = gemDigis->begin(); gemdgIt != gemDigis->end(); ++gemdgIt) {
      
    const GEMDetId& gemId = (*gemdgIt).first;
    
    std::map<int, std::vector<int> > vFatToStripMap;
    
    const GEMDigiCollection::Range& range = (*gemdgIt).second;
    for (GEMDigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt){
      const GEMDigi & digi = (*digiIt);
      //int bx    = digi.bx(); // setting all bx to 0 for now
      int strip = digi.strip();
      // use strip to get vFat ID
      // pair<int, int > vFatChan = vFatChannel(gemId, strip);
      // int vFatID = vFatChan.first
      // int chan   = vFatChan.second
      int vFatID = 1;
      int chan = strip*vFatID;

      vFatToStripMap[vFatID].push_back(chan);	
    }
    

    AMC13Event * amc13Event = new AMC13Event();
    AMCdata * amcData = new AMCdata();

    GEBdata * gebData = new GEBdata();

    std::map<int, std::vector<int> >::const_iterator vFatStrIt = vFatToStripMap.begin();    
    for (; vFatStrIt != vFatToStripMap.end(); ++vFatStrIt) {

      if (vFatStrIt->second.size() == 0) continue;
      
      uint8_t  b1010      =0xA;           ///<1010:4 Control bits, shoud be 1010
      uint16_t BC         =0;             ///<Bunch Crossing number, 12 bits
      uint8_t  b1100      =0xC;           ///<1100:4, Control bits, shoud be 1100
      uint8_t  EC         =0;             ///<Event Counter, 8 bits
      uint8_t  Flag       =0;             ///<Control Flags: 4 bits, Hamming Error/AFULL/SEUlogic/SUEI2C
      uint8_t  b1110      =0xE;           ///<1110:4 Control bits, shoud be 1110
      uint16_t crc        =0;             ///<Check Sum value, 16 bits
      uint16_t crc_calc   =0;             ///<Check Sum value recalculated, 16 bits
      int      SlotNumber =0;             ///<Calculated chip position
      bool     isBlockGood=0;             ///<Shows if block is good (control bits, chip ID and CRC checks)

      uint16_t ChipID = vFatStrIt->first; ///<Chip ID, 12 bits
      uint64_t lsData     =0;             ///<channels from 1to64 
      uint64_t msData     =0;             ///<channels from 65to128
      
      for (auto strip : vFatStrIt->second){
	std::cout <<"strip "<< strip<< std::endl;
	// set lsData and msData depending on strip no.	
      }
      
      VFATdata * vfatData =
	new VFATdata(b1010, BC, b1100, EC, Flag, b1110, ChipID, lsData, msData, crc, crc_calc, SlotNumber, isBlockGood);

      gebData->v_add(*vfatData);
      delete vfatData;
    }

    amcData->g_add(*gebData);
    delete gebData;

    amc13Event->addAMCpayload(*amcData);
    delete amcData;

    amc13Events.push_back(amc13Event);
    
  }

  for (auto amc13It : amc13Events){
    AMC13Event * amc13Event = amc13It;
    std::cout <<"amc13Event->nAMC() "<< int(amc13Event->nAMC()) << std::endl;
    
    //FEDRawData * rawData = new FEDRawData(amc13Event.dataSize());

    int fedId = FEDNumbering::MINGEMFEDID;    
    int dataSize = sizeof(uint64_t)*100;

    FEDRawData & fedRawData = fedRawDataCol->FEDData(fedId);
    fedRawData.resize(dataSize);
    
    uint64_t * word = reinterpret_cast<uint64_t* >(fedRawData.data());

    FEDHeader::set( fedRawData.data(), event_type_, e.id().event(), e.bunchCrossing(), fedId );
    word++;

    // // write data
    // unsigned int nWord32InFed = words.find(fedId)->second.size();
    // for (unsigned int i=0; i < nWord32InFed; i+=2) {
    //   *word = (Word64(words.find(fedId)->second[i]) << 32 ) | words.find(fedId)->second[i+1];
    //   LogDebug("PixelDataFormatter")  << print(*word);
    //   word++;
    // }

    // write one trailer
    FEDTrailer::set( fedRawData.data(), dataSize/sizeof(uint64_t), 0,0,0 );
    word++;
  }
  
  e.put(std::move(fedRawDataCol));
}
