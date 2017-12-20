/** \packer for me0
 *  \author J. Lee - UoS
 */
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
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

#include "EventFilter/GEMRawToDigi/plugins/ME0DigiToRawModule.h"

using namespace gem;

ME0DigiToRawModule::ME0DigiToRawModule(const edm::ParameterSet & pset):
  event_type_(pset.getParameter<int>("eventType")),
  digi_token(consumes<ME0DigiCollection>( pset.getParameter<edm::InputTag>("me0Digi") )),
  useDBEMap_(pset.getParameter<bool>("useDBEMap"))
{
  produces<FEDRawDataCollection>();
}

void ME0DigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("me0Digi", edm::InputTag("simMuonME0Digis"));
  desc.add<int>("eventType", 0);
  desc.add<bool>("useDBEMap", false);
  descriptions.add("me0Packer", desc);  
}

std::shared_ptr<ME0ROmap> ME0DigiToRawModule::globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const
{
  if (useDBEMap_){
    edm::ESHandle<ME0EMap> me0EMapRcd;
    iSetup.get<ME0EMapRcd>().get(me0EMapRcd);
    auto me0EMap = std::make_unique<ME0EMap>(*(me0EMapRcd.product()));
    return std::make_shared<ME0ROmap>(*(me0EMap->convert()));
  }
  else {
    // no eMap, using dummy
    auto me0EMap = std::make_unique<ME0EMap>();
    return std::make_shared<ME0ROmap>(*(me0EMap->convertDummy()));
  }
}

void ME0DigiToRawModule::produce(edm::StreamID iID, edm::Event & iEvent, edm::EventSetup const&) const
{
  auto fedRawDataCol = std::make_unique<FEDRawDataCollection>();

  // Take digis from the event
  edm::Handle<ME0DigiCollection> me0Digis;
  iEvent.getByToken( digi_token, me0Digis );
  
  auto me0ROMap = runCache(iEvent.getRun().index());
  
  std::vector<std::unique_ptr<AMC13Event>> amc13Events;
  // currently only one FEDRaw
  amc13Events.reserve(1);
  {
    auto amc13Event = std::make_unique<AMC13Event>();
    
    for (const auto & amc : me0ROMap->getAMCs()){	
      auto amcData = std::make_unique<AMCdata>();
      uint16_t amcId = amc;

      for (const auto & geb : me0ROMap->getAMC2GEBs(amcId)){
	
	uint16_t gebId = geb;
	uint32_t chamberId = (amcId << 5) | gebId;
	
	ME0DetId chamDetId = me0ROMap->gebPosition(chamberId);

	auto gebData = std::make_unique<GEBdata>();
	gebData->setInputID(gebId);
	
	// 1 GEB per chamber
	// making vfats
	for (uint16_t bc = 0; bc < 50; ++bc){
	  std::unordered_map<uint32_t, std::vector<int> > vFatToStripMap;
	  bool hasDigi = false;
	
	  for (int roll = 1; roll<=ME0DetId::maxRollId; ++roll){

	    ME0DetId me0Id(chamDetId.region(), chamDetId.layer(), chamDetId.chamber(), roll);
	    	  
	    ME0DigiCollection::Range range = me0Digis->get(me0Id);
	    for (ME0DigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt){

	      const ME0Digi & digi = (*digiIt);
	      if (digi.bx() != bc-25) continue;
	
	      ME0ROmap::dCoord dc;
	      dc.me0DetId = me0Id;
	      dc.stripId = digi.strip();

	      ME0ROmap::eCoord ec = me0ROMap->hitPosition(dc);
	      uint32_t vFatID = ec.vfatId;
	      int channelId = ec.channelId;
		
	      vFatToStripMap[vFatID].emplace_back(channelId);	
	      hasDigi = true;

	      LogDebug("ME0DigiToRawModule") <<" vfatId "<<ec.vfatId
					     <<" me0DetId "<< me0Id
					     <<" chan "<< ec.channelId
					     <<" strip "<< dc.stripId
					     <<" bx "<< digi.bx();
	    }
	  }
	  
	  if (!hasDigi) continue;
	  
	  // fill in vFat
	  for( const auto& vPair : vFatToStripMap ) {
	    if (vPair.second.empty()) continue;
      
	    uint8_t  b1010      =0xA;           ///<1010:4 Control bits, shoud be 1010
	    uint16_t BC         =bc;            ///<Bunch Crossing number, 12 bits
	    uint8_t  b1100      =0xC;           ///<1100:4, Control bits, shoud be 1100
	    uint8_t  EC         =0;             ///<Event Counter, 8 bits
	    uint8_t  Flag       =0;             ///<Control Flags: 4 bits, Hamming Error/AFULL/SEUlogic/SUEI2C
	    uint8_t  b1110      =0xE;           ///<1110:4 Control bits, shoud be 1110
	    int      SlotNumber =0;             ///<Calculated chip position
	    bool     isBlockGood=false;         ///<Shows if block is good (control bits, chip ID and CRC checks)
	    uint16_t ChipID = 0xFFF & vPair.first; ///<Chip ID, 12 bits
	    uint64_t lsData     =0;             ///<channels from 1to64 
	    uint64_t msData     =0;             ///<channels from 65to128
	    uint16_t crc        =0;             ///<Check Sum value, 16 bits
	    uint16_t crc_calc   =0;             ///<Check Sum value recalculated, 16 bits
	  
	    for (const auto & chan : vPair.second){
	      uint64_t oneBit = 0x1;
	      if (chan < 64) lsData = lsData | (oneBit << chan);
	      else msData = msData | (oneBit << (chan-64));
	    }
	    
	    auto vfatData = std::make_unique<VFATdata>(b1010, BC, b1100, EC, Flag, b1110, ChipID, lsData, msData,
						       crc, crc_calc, SlotNumber, isBlockGood);
	    
	    gebData->addVFAT(*vfatData);
	  }
	}
	
	if (!gebData->vFATs().empty()){
	  gebData->setInputID(gebId);
	  gebData->setVwh(gebData->vFATs().size()*3);
	  amcData->addGEB(*gebData);
	}
      }

      if (!amcData->gebs().empty()){
	amcData->setGDcount(amcData->gebs().size());
	amcData->setBID(amcId);
	amc13Event->addAMCpayload(*amcData);
      }
    }

    // CDFHeader
    uint8_t cb5 = 0x5;// control bit, should be 0x5 bits 60-63
    uint8_t Evt_ty = event_type_;
    uint32_t LV1_id = iEvent.id().event();
    uint16_t BX_id = iEvent.bunchCrossing();
    uint16_t Source_id = FEDNumbering::MINME0FEDID;
    amc13Event->setCDFHeader(cb5, Evt_ty, LV1_id, BX_id, Source_id);

    // AMC13header
    uint8_t CalTyp = 1;
    uint8_t nAMC = amc13Event->getAMCpayload().size(); // currently only one AMC13Event
    uint32_t OrN = 2;
    uint8_t cb0  = 0b0000;// control bit, should be 0b0000
    amc13Event->setAMC13header(CalTyp, nAMC, OrN, cb0);

    for (unsigned short i = 0; i < amc13Event->nAMC(); ++i){
      uint32_t AMC_size = 0;
      uint8_t Blk_No = 0;
      uint8_t AMC_No = 0;
      uint16_t BoardID = 0;
      amc13Event->addAMCheader(AMC_size, Blk_No, AMC_No, BoardID);
    }
    
    //AMC13 trailer
    uint32_t CRC_amc13 = 0;
    uint8_t Blk_NoT = 0;
    uint8_t LV1_idT = 0;
    uint16_t BX_idT = BX_id;
    amc13Event->setAMC13trailer(CRC_amc13, Blk_NoT, LV1_idT, BX_idT);
    //CDF trailer
    uint8_t cbA = 0xA; // control bit, should be 0xA bits 60-63
    uint32_t EvtLength = 0;
    uint16_t CRC_cdf = 0;
    amc13Event->setCDFTrailer(cbA, EvtLength, CRC_cdf);  
    amc13Events.emplace_back(std::move(amc13Event));
  }

  
  // read out amc13Events into fedRawData
  for (const auto & amc13e : amc13Events){
    std::vector<uint64_t> words;    
    words.emplace_back(amc13e->getCDFHeader());
    words.emplace_back(amc13e->getAMC13header());    

    for (const auto & w: amc13e->getAMCheader())
      words.emplace_back(w);    

    for (const auto & amc : amc13e->getAMCpayload()){      
      words.emplace_back(amc.getAMCheader1());      
      words.emplace_back(amc.getAMCheader2());
      words.emplace_back(amc.getGEMeventHeader());

      for (const auto & geb: amc.gebs()){
	words.emplace_back(geb.getChamberHeader());

	for (const auto & vfat: geb.vFATs()){
	  words.emplace_back(vfat.get_fw());
	  words.emplace_back(vfat.get_sw());
	  words.emplace_back(vfat.get_tw());
	}
	
	words.emplace_back(geb.getChamberTrailer());
      }
      
      words.emplace_back(amc.getGEMeventTrailer());
      words.emplace_back(amc.getAMCTrailer());
    }
    
    words.emplace_back(amc13e->getAMC13trailer());
    words.emplace_back(amc13e->getCDFTrailer());

    FEDRawData & fedRawData = fedRawDataCol->FEDData(amc13e->source_id());
    
    int dataSize = (words.size()) * sizeof(uint64_t);
    fedRawData.resize(dataSize);
    
    uint64_t * w = reinterpret_cast<uint64_t* >(fedRawData.data());  
    for (const auto & word: words) *(w++) = word;
    
    LogDebug("ME0DigiToRawModule") <<" words " << words.size();
  }

  iEvent.put(std::move(fedRawDataCol));
}
