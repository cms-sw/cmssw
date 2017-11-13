/** \packer for me0
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

#include "EventFilter/GEMRawToDigi/plugins/ME0DigiToRawModule.h"

using namespace gem;

ME0DigiToRawModule::ME0DigiToRawModule(const edm::ParameterSet & pset) 
{
  event_type_ = pset.getParameter<int>("eventType");
  digi_token = consumes<ME0DigiCollection>( pset.getParameter<edm::InputTag>("me0Digi") );
  useDBEMap_ = pset.getParameter<bool>("useDBEMap");
  produces<FEDRawDataCollection>();
}

void ME0DigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("me0Digi", edm::InputTag("simMuonME0Digis"));
}

void ME0DigiToRawModule::doBeginRun_(edm::Run const& rp, edm::EventSetup const& iSetup)
{
  if (useDBEMap_){
    edm::ESHandle<ME0EMap> me0EMap;
    iSetup.get<ME0EMapRcd>().get(me0EMap);
    m_me0EMap = me0EMap.product();
    m_me0ROMap = m_me0EMap->convert();
  }
  else {
    // no eMap, using dummy
    m_me0EMap = new ME0EMap();
    m_me0ROMap = m_me0EMap->convertDummy();
  }
}

void ME0DigiToRawModule::produce(edm::StreamID, edm::Event & e, const edm::EventSetup & iSetup) const
{
  auto fedRawDataCol = std::make_unique<FEDRawDataCollection>();

  // Take digis from the event
  edm::Handle<ME0DigiCollection> me0Digis;
  e.getByToken( digi_token, me0Digis );

  std::vector<AMC13Event*> amc13Events;
  
  // currently only one FEDRaw
  {
    AMC13Event * amc13Event = new AMC13Event();
    
    for (auto amc : m_me0ROMap->getAMCs()){	
      AMCdata * amcData = new AMCdata();
      uint16_t amcId = amc;

      for (auto geb : m_me0ROMap->getAMC2GEBs(amcId)){
	
	uint16_t gebId = geb;
	uint32_t chamberId = (amcId << 5) | gebId;
	
	ME0DetId chamDetId = m_me0ROMap->gebPosition(chamberId);

	GEBdata * gebData = new GEBdata();
	gebData->setInputID(gebId);
	
	// 1 GEB per chamber
	// making vfats
	for (uint16_t bc = 0; bc < 50; ++bc){
	  std::map<int, std::vector<int> > vFatToStripMap;
	  bool hasDigi = false;
	
	  for (int roll = 1; roll<=8; ++roll){
	  
	    ME0DetId me0Id(chamDetId.region(), chamDetId.layer(), chamDetId.chamber(), roll);
	  
	    ME0DigiCollection::Range range = me0Digis->get(me0Id);
	    for (ME0DigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt){

	      const ME0Digi & digi = (*digiIt);
	      if (digi.bx() != bc-25) continue;
	
	      ME0ROmap::dCoord dc;
	      dc.me0DetId = me0Id;
	      dc.stripId = digi.strip();

	      ME0ROmap::eCoord ec = m_me0ROMap->hitPosition(dc);
	      uint32_t vFatID = ec.vfatId;
	      int channelId = ec.channelId;
		
	      vFatToStripMap[vFatID].push_back(channelId);	
	      hasDigi = true;

	      // std::cout <<"ME0DigiToRawModule vfatId "<<ec.vfatId
	      // 		<<" me0DetId "<< me0Id
	      // 		<<" chan "<< ec.channelId
	      // 		<<" strip "<< dc.stripId
	      // 		<<" bx "<< digi.bx()
	      // 		<<std::endl;
	    }
	  }
	  
	  if (!hasDigi) continue;
	  
	  // fill in vFat
	  std::map<int, std::vector<int> >::const_iterator vFatStrIt = vFatToStripMap.begin();    
	  for (; vFatStrIt != vFatToStripMap.end(); ++vFatStrIt) {

	    if (vFatStrIt->second.empty()) continue;
      
	    uint8_t  b1010      =0xA;           ///<1010:4 Control bits, shoud be 1010
	    uint16_t BC         =bc;            ///<Bunch Crossing number, 12 bits
	    uint8_t  b1100      =0xC;           ///<1100:4, Control bits, shoud be 1100
	    uint8_t  EC         =0;             ///<Event Counter, 8 bits
	    uint8_t  Flag       =0;             ///<Control Flags: 4 bits, Hamming Error/AFULL/SEUlogic/SUEI2C
	    uint8_t  b1110      =0xE;           ///<1110:4 Control bits, shoud be 1110
	    uint16_t crc        =0;             ///<Check Sum value, 16 bits
	    uint16_t crc_calc   =0;             ///<Check Sum value recalculated, 16 bits
	    int      SlotNumber =0;             ///<Calculated chip position
	    bool     isBlockGood=true;          ///<Shows if block is good (control bits, chip ID and CRC checks)

	    uint16_t ChipID = 0xFFF & vFatStrIt->first; ///<Chip ID, 12 bits
	    uint64_t lsData     =0;             ///<channels from 1to64 
	    uint64_t msData     =0;             ///<channels from 65to128
	  
	    for (auto chan : vFatStrIt->second){
	      uint64_t oneBit = 0x1;
	      if (chan < 64) lsData = lsData | (oneBit << chan);
	      else msData = msData | (oneBit << (chan-64));
	    }
	    
	    VFATdata * vfatData =
	      new VFATdata(b1010, BC, b1100, EC, Flag, b1110, ChipID, lsData, msData,
			   crc, crc_calc, SlotNumber, isBlockGood);
	    
	    
	    gebData->v_add(*vfatData);
	    delete vfatData;
	  }
	}
	
	if (!gebData->vfats().empty()){
	  gebData->setInputID(gebId);
	  gebData->setVwh(gebData->vfats().size()*3);
	  amcData->g_add(*gebData);
	}
	delete gebData;	
      }

      if (!amcData->gebs().empty()){
	amcData->setGDcount(amcData->gebs().size());
	amcData->setBID(amcId);
	amc13Event->addAMCpayload(*amcData);
      }
      delete amcData;
    }

    // CDFHeader
    uint8_t cb5 = 0x5;// control bit, should be 0x5 bits 60-63
    uint8_t Evt_ty = event_type_;
    uint32_t LV1_id = e.id().event();
    uint16_t BX_id = e.bunchCrossing();
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
    amc13Events.push_back(amc13Event);    
  }

  
  // read out amc13Events into fedRawData
  for (auto amc13It : amc13Events){
    AMC13Event * amc13Event = amc13It;
    std::vector<uint64_t> words;    
    words.push_back(amc13Event->getCDFHeader());
    words.push_back(amc13Event->getAMC13header());    

    for (auto w: amc13Event->getAMCheader())
      words.push_back(w);    

    for (auto amc : amc13Event->getAMCpayload()){
      AMCdata * amcData = &amc;
      
      words.push_back(amcData->getAMCheader1());      
      words.push_back(amcData->getAMCheader2());
      words.push_back(amcData->getGEMeventHeader());

      for (auto geb: amcData->gebs()){
	GEBdata * gebData = &geb;
	words.push_back(gebData->getChamberHeader());

	for (auto vfat: gebData->vfats()){
	  VFATdata * vfatData = &vfat;
	  words.push_back(vfatData->get_fw());
	  words.push_back(vfatData->get_sw());
	  words.push_back(vfatData->get_tw());
	  //delete vfatData;
	}
	
	words.push_back(gebData->getChamberTrailer());
	//delete gebData;
      }
      
      words.push_back(amcData->getGEMeventTrailer());
      words.push_back(amcData->getAMCTrailer());
      //delete amcData;
    }
    
    words.push_back(amc13Event->getAMC13trailer());
    words.push_back(amc13Event->getCDFTrailer());

    FEDRawData & fedRawData = fedRawDataCol->FEDData(amc13Event->Source_id());
    
    int dataSize = (words.size()) * sizeof(uint64_t);
    fedRawData.resize(dataSize);
    
    uint64_t * w = reinterpret_cast<uint64_t* >(fedRawData.data());  
    for (auto word: words) *(w++) = word;
        
    //std::cout << "ME0DigiToRawModule words " <<std::dec << words.size() << std::endl;
    delete amc13Event;
  }

  e.put(std::move(fedRawDataCol));
}
