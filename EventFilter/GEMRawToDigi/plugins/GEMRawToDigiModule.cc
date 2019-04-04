/** \unpacker for gem
 *  \author J. Lee - UoS
 */
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/GEMRawToDigi/plugins/GEMRawToDigiModule.h"

using namespace gem;

GEMRawToDigiModule::GEMRawToDigiModule(const edm::ParameterSet & pset) :
  fed_token(consumes<FEDRawDataCollection>( pset.getParameter<edm::InputTag>("InputLabel") )),
  useDBEMap_(pset.getParameter<bool>("useDBEMap")),
  unPackStatusDigis_(pset.getParameter<bool>("unPackStatusDigis"))
{
  produces<GEMDigiCollection>(); 
  if (unPackStatusDigis_) {
    produces<GEMVfatStatusDigiCollection>("vfatStatus");
    produces<GEMGEBdataCollection>("gebStatus");
    produces<GEMAMCdataCollection>("AMCdata"); 
    produces<GEMAMC13EventCollection>("AMC13Event"); 
  }
}

void GEMRawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector")); 
  desc.add<bool>("useDBEMap", false);
  desc.add<bool>("unPackStatusDigis", false);
  descriptions.add("muonGEMDigisDefault", desc);  
}

std::shared_ptr<GEMROMapping> GEMRawToDigiModule::globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const
{
  auto gemROmap = std::make_shared<GEMROMapping>();
  if (useDBEMap_) {
    edm::ESHandle<GEMeMap> gemEMapRcd;
    iSetup.get<GEMeMapRcd>().get(gemEMapRcd);
    auto gemEMap = std::make_unique<GEMeMap>(*(gemEMapRcd.product()));
    gemEMap->convert(*gemROmap);
    gemEMap.reset();    
  }
  else {
    // no EMap in DB, using dummy
    auto gemEMap = std::make_unique<GEMeMap>();
    gemEMap->convertDummy(*gemROmap);
    gemEMap.reset();    
  }
  return gemROmap;
}

void GEMRawToDigiModule::produce(edm::StreamID iID, edm::Event & iEvent, edm::EventSetup const&) const
{
  auto outGEMDigis = std::make_unique<GEMDigiCollection>();
  auto outVFATStatus = std::make_unique<GEMVfatStatusDigiCollection>();
  auto outGEBStatus = std::make_unique<GEMGEBdataCollection>();
  auto outAMCdata = std::make_unique<GEMAMCdataCollection>();
  auto outAMC13Event = std::make_unique<GEMAMC13EventCollection>();

  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  iEvent.getByToken( fed_token, fed_buffers );
  
  auto gemROMap = runCache(iEvent.getRun().index());
  
  for (unsigned int fedId=FEDNumbering::MINGEMFEDID; fedId<=FEDNumbering::MAXGEMFEDID; ++fedId) { 
    const FEDRawData& fedData = fed_buffers->FEDData(fedId);
    
    int nWords = fedData.size()/sizeof(uint64_t);
    LogDebug("GEMRawToDigiModule") <<" words " << nWords;
    
    if (nWords<5) continue;
    const unsigned char * data = fedData.data();
    
    auto amc13Event = std::make_unique<AMC13Event>();
    
    const uint64_t* word = reinterpret_cast<const uint64_t* >(data);
    
    amc13Event->setCDFHeader(*word);
    amc13Event->setAMC13Header(*(++word));
    
    // Readout out AMC headers
    for (uint8_t i = 0; i < amc13Event->nAMC(); ++i)
      amc13Event->addAMCheader(*(++word));
    
    // Readout out AMC payloads
    for (uint8_t i = 0; i < amc13Event->nAMC(); ++i) {
      auto amcData = std::make_unique<AMCdata>();
      amcData->setAMCheader1(*(++word));      
      amcData->setAMCheader2(*(++word));
      amcData->setGEMeventHeader(*(++word));
      uint16_t amcBx = amcData->bx();
      uint8_t amcNum = amcData->amcNum();

      // Fill GEB
      for (uint8_t j = 0; j < amcData->davCnt(); ++j) {
	auto gebData = std::make_unique<GEBdata>();
	gebData->setChamberHeader(*(++word));
	
	uint8_t gebId = gebData->inputID();
	GEMROMapping::chamEC geb_ec = {fedId, amcNum, gebId};        
	GEMROMapping::chamDC geb_dc = gemROMap->chamberPos(geb_ec);
	GEMDetId gemChId = geb_dc.detId;

	for (uint16_t k = 0; k < gebData->vfatWordCnt()/3; k++) {
	  auto vfatData = std::make_unique<VFATdata>();
	  vfatData->read_fw(*(++word));
	  vfatData->read_sw(*(++word));
	  vfatData->read_tw(*(++word));
	  
          vfatData->setVersion(geb_dc.vfatVer);
          uint16_t vfatId = vfatData->vfatId();
          GEMROMapping::vfatEC vfat_ec = {vfatId, gemChId};
          
	  // check if ChipID exists.
	  if (!gemROMap->isValidChipID(vfat_ec)) {
	    edm::LogWarning("GEMRawToDigiModule") << "InValid: amcNum "<< int(amcNum)
						  << " gebId "<< int(gebId)
						  << " vfatId "<< int(vfatId)
						  << " vfat Pos "<< int(vfatData->position());
	    continue;
	  }
          // check vfat data
	  if (vfatData->quality()) {
	    edm::LogWarning("GEMRawToDigiModule") << "Quality "<< int(vfatData->quality())
						  << " b1010 "<< int(vfatData->b1010())
						  << " b1100 "<< int(vfatData->b1100())
						  << " b1110 "<< int(vfatData->b1110());
	    if (vfatData->crc() != vfatData->checkCRC() ) {
	      edm::LogWarning("GEMRawToDigiModule") << "DIFFERENT CRC :"
						    <<vfatData->crc()<<"   "<<vfatData->checkCRC();	      
	    }
	  }
	            
          GEMROMapping::vfatDC vfat_dc = gemROMap->vfatPos(vfat_ec);

	  vfatData->setPhi(vfat_dc.localPhi);
          GEMDetId gemId = vfat_dc.detId;
	  uint16_t bc=vfatData->bc();
	  // strip bx = vfat bx - amc bx
	  int bx = bc-amcBx;
          
	  for (int chan = 0; chan < VFATdata::nChannels; ++chan) {
	    uint8_t chan0xf = 0;
	    if (chan < 64) chan0xf = ((vfatData->lsData() >> chan) & 0x1);
	    else chan0xf = ((vfatData->msData() >> (chan-64)) & 0x1);

	    // no hits
	    if (chan0xf==0) continue;
	    	             
            GEMROMapping::channelNum chMap = {vfat_dc.vfatType, chan};
            GEMROMapping::stripNum stMap = gemROMap->hitPos(chMap);

            int stripId = stMap.stNum + vfatData->phi()*GEMeMap::maxChan_;    
            
	    GEMDigi digi(stripId,bx);

            LogDebug("GEMRawToDigiModule")
              << " fed: " << fedId
              << " amc:" << int(amcNum)
              << " geb:" << int(gebId)
              << " vfat:"<< vfat_dc.localPhi
              << ",type: "<< vfat_dc.vfatType
              << " id:"<< gemId
              << " ch:"<< chMap.chNum
              << " st:"<< digi.strip()
              << " bx:"<< digi.bx();
            
	    outGEMDigis.get()->insertDigi(gemId,digi);
	    
	  }// end of channel loop
	  
	  if (unPackStatusDigis_) {
            outVFATStatus.get()->insertDigi(gemId, GEMVfatStatusDigi(*vfatData));
	  }

	} // end of vfat loop
	
	gebData->setChamberTrailer(*(++word));
	
        if (unPackStatusDigis_) {
	  outGEBStatus.get()->insertDigi(gemChId.chamberId(), (*gebData)); 
        }
	
      } // end of geb loop
      
      amcData->setGEMeventTrailer(*(++word));
      amcData->setAMCTrailer(*(++word));
      
      if (unPackStatusDigis_) {
        outAMCdata.get()->insertDigi(amcData->boardId(), (*amcData));
      }

    } // end of amc loop
    
    amc13Event->setAMC13Trailer(*(++word));
    amc13Event->setCDFTrailer(*(++word));
     
    if (unPackStatusDigis_) {
      outAMC13Event.get()->insertDigi(amc13Event->bxId(), AMC13Event(*amc13Event));
    }
    
  } // end of amc13Event
  
  iEvent.put(std::move(outGEMDigis));
  
  if (unPackStatusDigis_) {
    iEvent.put(std::move(outVFATStatus), "vfatStatus");
    iEvent.put(std::move(outGEBStatus), "gebStatus");
    iEvent.put(std::move(outAMCdata), "AMCdata");
    iEvent.put(std::move(outAMC13Event), "AMC13Event");
  }
  
}
