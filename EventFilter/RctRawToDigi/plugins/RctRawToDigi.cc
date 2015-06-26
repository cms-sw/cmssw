#include "EventFilter/RctRawToDigi/plugins/RctRawToDigi.h"

// System headers
#include <vector>
#include <sstream>
#include <iostream>

// Framework headers
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// Raw data collection headers
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"
#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"


// Namespace resolution
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::dec;
using std::hex;
using edm::LogWarning;
using edm::LogError;

#define EVENT_HEADER_WORDS 6
#define CHANNEL_HEADER_WORDS 2
#define CHANNEL_DATA_WORDS_PER_BX 6
#define NIntsBRAMDAQ 1024*2
#define slinkHeaderSize_ 8
#define slinkTrailerSize_ 8
#define amc13HeaderSize_ 8
#define amc13TrailerSize_ 8
#define NLinks 36

RctRawToDigi::RctRawToDigi(const edm::ParameterSet& iConfig) :
  inputLabel_(iConfig.getParameter<edm::InputTag>("inputLabel")),
  fedId_(iConfig.getUntrackedParameter<int>("rctFedId", FEDNumbering::MINTriggerUpgradeFEDID)),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose",false))
{
  LogDebug("RCT") << "RctRawToDigi will unpack FED Id " << fedId_;
  std::cout<< "RctRawToDigi will unpack FED Id " << fedId_<<std::endl;
  /** Register Products **/
  // RCT collections
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();

  // Error collection
  consumes<FEDRawDataCollection>(inputLabel_);
  std::cout<<"finishde initialization"<<std::endl;
}


RctRawToDigi::~RctRawToDigi()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //delete formatTranslator_;
}

// ------------ method called to produce the data  ------------
void RctRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::cout<<"Produce"<<std::endl;
  using namespace edm;

  // Instantiate all the collections the unpacker needs; puts them in event when this object goes out of scope.
  std::auto_ptr<RctUnpackCollections> colls(new RctUnpackCollections(iEvent));  
  
  // get raw data collection
  edm::Handle<FEDRawDataCollection> feds;
  iEvent.getByLabel(inputLabel_, feds);
  //colls()->gctHfBitCounts()->push_back( CONTINUE HERE

  // if raw data collection is present, check the headers and do the unpacking
  if (feds.isValid()) {
  
    const FEDRawData& rctRcd = feds->FEDData(fedId_);

    //Check FED size
    LogDebug("RCT") << "Upacking FEDRawData of size " << std::dec << rctRcd.size();

    //try to check slink header size
    if ((int) rctRcd.size() < slinkHeaderSize_ + slinkTrailerSize_ + amc13HeaderSize_ + amc13TrailerSize_ ) {
      LogError("L1T") << "Cannot unpack: empty/invalid L1T raw data (size = "
		      << rctRcd.size() << ") for ID " << fedId_ << ". Returning empty collections!";
      //continue;
      return;
    }

    // do the unpacking
    unpack(rctRcd, iEvent, colls.get()); 

    // dump summary in verbose mode
    if(verbose_) { doVerboseOutput(colls.get()); }
    
  }
  else{

    LogError("L1T") << "Cannot unpack: no collection found";

    return; 
  }

}

void RctRawToDigi::unpack(const FEDRawData& d, edm::Event& e, RctUnpackCollections * const colls)
{

  // We should now have a valid formatTranslator pointer
  //formatTranslator_->setUnpackCollections(colls);
  std::cout<<"Unpacking"<<std::endl;
  const unsigned char * data = d.data();  // The 8-bit wide raw-data array.  

  // Data offset - starts at 16 as there is a 64-bit S-Link header followed
  // by a 64-bit software-controlled header (for pipeline format version
  // info that is not yet used).
  //unsigned dPtr = 16;

  //const unsigned dEnd = d.size() - 8; // End of payload is at (packet size - final slink header)
  ///Do unpacking here 
  
  FEDHeader header(data);
  
  if (header.check()) {
    LogDebug("L1T") << "Found SLink header:"
		    << " Trigger type " << header.triggerType()
		    << " L1 event ID " << header.lvl1ID()
		    << " BX Number " << header.bxID()
		    << " FED source " << header.sourceID()
		    << " FED version " << header.version();
  } else {
    LogWarning("L1T") << "Did not find a valid SLink header!";
  }
  
  FEDTrailer trailer(data + (d.size() - slinkTrailerSize_));
  
  if (trailer.check()) {
    LogDebug("L1T") << "Found SLink trailer:"
		    << " Length " << trailer.lenght()
		    << " CRC " << trailer.crc()
		    << " Status " << trailer.evtStatus()
		    << " Throttling bits " << trailer.ttsBits();
  } else {
    LogWarning("L1T") << "Did not find a SLink trailer!";
  }
  
  //amc13::Packet packet;
  //if (!packet.parse((const uint64_t*) (data + slinkHeaderSize_),
  //		    (d.size() - slinkHeaderSize_ - slinkTrailerSize_) / 8)) {
  //LogError("L1T") << "Could not extract AMC13 Packet.";
  //return;
  //}

  //std::unique_ptr<uint64_t[]> payload64 = amc.data();
  //const uint32_t * start = (const uint32_t*) payload64.get();
  //const uint32_t * end = start + (amc.size() * 2);
  
  //std::auto_ptr<Payload> payload;
  
  //unsigned fw = payload->getFirmwareId();
  
  // Let parameterset value override FW version
  //if (fwId_ > 0)
  //fw = fwId_;
  
  //unsigned board = amc.header().getBoardID();
  //unsigned amc_no = amc.header().getAMCNumber();
	 
  //unpackCTP7(data, block_id, data.size(), colls)
  //d or data?
  printAll(data, d.size());

}




void RctRawToDigi::doVerboseOutput(const RctUnpackCollections * const colls) const
{
  std::ostringstream os;

  //for (unsigned i=0, size = bHdrs.size(); i < size; ++i)
  // {
  //os << "RCT Raw Data Block : " << formatTranslator_->getBlockDescription(bHdrs[i]) << " : " << bHdrs[i] << endl;
  //}
  //os << *colls << endl;
  //edm::LogVerbatim("RCT") << os.str();
}



void
RctRawToDigi::unpackCTP7(const unsigned char *data, const unsigned block_id, const unsigned size, RctUnpackCollections * const colls)
{

     LogDebug("L1T") << "Block ID  = " << block_id << " size = " << size;

     RctDataDecoder rctDataDecoder;
     uint32_t nBX = 0; 
     uint32_t ctp7FWVersion;
     uint32_t L1ID, L1aBCID;
     
     L1ID          =  data[1];                      // extract the L1 ID number
     L1aBCID       =  data[5] & 0x00000FFF;         // extract the BCID number of L1A
     nBX           = (data[5] & 0x00FF0000) >> 16;  // extract number of BXs readout per L1A 
     ctp7FWVersion =  data[4];
     
     LogDebug("L1T") << "L1ID = " << L1ID << " L1A BCID = " << L1aBCID << " BXs in capture = " << nBX << " CTP7 DAQ FW = " << ctp7FWVersion;

     //getBXNumbers handles special cases, for example:
     //if L1A is 3563, nBX = 3 then BCs = 3562, 3563, 0
     //uint32_t firstBX = 0; uint32_t lastBX = 1; 

     std::vector<RCTInfo> allCrateRCTInfo[nBX];

     struct link_data{
       unsigned int ctp7LinkNumber;
       //full 32-bit linkID
       unsigned int capturedLinkID;
       //decoded linkID from the oRSC
       unsigned int capturedLinkNumber;
       bool even;
       unsigned int crateID;
       std::vector <unsigned int> uint;
     };

     link_data allLinks[nBX][NLinks];

     //change this implementation
     uint32_t iDAQBuffer = 0;

     //Step 1: Grab all data from ctp7 buffer and put into link format
     for(unsigned int iLink = 0; iLink < NLinks; iLink++ ){
       
       iDAQBuffer = EVENT_HEADER_WORDS + iLink * (CHANNEL_HEADER_WORDS + 
						  nBX * CHANNEL_DATA_WORDS_PER_BX);
       
       //first decode linkID 
       uint32_t linkID     = data[iDAQBuffer++];//pop here?
       uint32_t tmp        = data[iDAQBuffer++];
       uint32_t CRCErrCnt  =  tmp & 0x0000FFFF;
       uint32_t linkStatus = (tmp & 0xFFFF0000) >> 16;
       
       if(verbose_) LogDebug("L1T")<< std::hex<< "linkID "<< linkID << " CRCErrCnt " << CRCErrCnt << " linkStatus "<< linkStatus;
       
       uint32_t capturedLinkID = 0;
       uint32_t crateID = 0;
       uint32_t capturedLinkNumber = 0;
       bool even = false;

       //Using normal method for decoding oRSC Captured LinkID
       //Input linkID, output: crateID, capturedLinkNumber, Even or Odd
       decodeLinkID(linkID, crateID, capturedLinkNumber, even);
            
       // Loop over multiple BX                                                                        
       for (uint32_t iBX=0; iBX<nBX; iBX++){
	 allLinks[iLink][iBX].uint.reserve(6);
	 allLinks[iLink][iBX].ctp7LinkNumber     = iLink;
	 allLinks[iLink][iBX].capturedLinkID     = capturedLinkID;
	 allLinks[iLink][iBX].crateID            = crateID;
	 allLinks[iLink][iBX].capturedLinkNumber = capturedLinkNumber;
	 allLinks[iLink][iBX].even               = even;
	 
	 for(unsigned int iWord = 0; iWord < 6 ; iWord++ ){
	   allLinks[iLink][iBX].uint.push_back(data[iDAQBuffer+iWord]);
	 }
       }
     }
     
     //Step 2: Dynamically match links and create RCTInfo Objects 
     uint32_t nCratesFound = 0;
     for(unsigned int iCrate = 0; iCrate < 18 ; iCrate++){
       
       bool foundEven = false, foundOdd = false;
       link_data even[nBX];
       link_data odd[nBX];
       
       for(unsigned int iLink = 0; iLink < NLinks; iLink++){
	 
	 if( (allLinks[iLink][0].crateID==iCrate) && (allLinks[iLink][0].even == true) ){
	   
	   foundEven = true;
	   for (unsigned int iBX=0; iBX<nBX; iBX++)
	     even[iBX] = allLinks[iLink][iBX];
	     
	 }
	 else if( (allLinks[iLink][0].crateID==iCrate) && (allLinks[iLink][0].even == false) ){
	   
	   foundOdd = true;
	   for (unsigned int iBX=0; iBX<nBX; iBX++)
	     odd[iBX] = allLinks[iLink][iBX];
	   
	 } 
	 //if success then create RCTInfoVector and fill output object
	 if(foundEven && foundOdd){
	   nCratesFound++;
	   
	   //fill rctInfoVector for all BX read out
	   for (unsigned int iBX=0; iBX<nBX; iBX++){
	     std::vector <RCTInfo> rctInfoData;
	     rctDataDecoder.decodeLinks(even[iBX].uint, odd[iBX].uint, rctInfoData);
	     allCrateRCTInfo[iBX].push_back(rctInfoData.at(0));
	   }
	   break;
	 }
       }
     }
     
     if(nCratesFound != 18)
       LogDebug("L1T") << "Warning -- only found "<< nCratesFound << " valid crates";
   
     //Step 3: Create Collections from RCTInfo Objects  
     for (uint32_t iBX=0; iBX<nBX; iBX++){
       for(unsigned int iCrate = 0; iCrate < nCratesFound; iCrate++ ){

	 RCTInfo rctInfo = allCrateRCTInfo[iBX].at(iCrate);
	 //Use Crate ID to identify eta/phi of candidate
	 for(int j = 0; j < 4; j++) {
	   L1CaloEmCand em = L1CaloEmCand(rctInfo.neRank[j], 
					  rctInfo.neRegn[j], 
					  rctInfo.neCard[j], 
					  rctInfo.crateID, 
					  false);
	   em.setBx(iBX);
	   colls->rctEm()->push_back(em);
	 }

	 for(int j = 0; j < 4; j++) {
	   L1CaloEmCand em = L1CaloEmCand(rctInfo.ieRank[j], 
					  rctInfo.ieRegn[j], 
					  rctInfo.ieCard[j], 
					  rctInfo.crateID, 
					  true);
	   em.setBx(iBX);
	   colls->rctEm()->push_back(em);
	 }

	 for(int j = 0; j < 7; j++) {
	   for(int k = 0; k < 2; k++) {
	     bool o = (((rctInfo.oBits >> (j * 2 + k)) & 0x1) == 0x1);
	     bool t = (((rctInfo.tBits >> (j * 2 + k)) & 0x1) == 0x1);
	     bool m = (((rctInfo.mBits >> (j * 2 + k)) & 0x1) == 0x1);
	     bool q = (((rctInfo.qBits >> (j * 2 + k)) & 0x1) == 0x1);
	     L1CaloRegion rgn = L1CaloRegion(rctInfo.rgnEt[j][k], o, t, m, q, rctInfo.crateID , j, k);
	     rgn.setBx(iBX);
	     colls->rctCalo()->push_back(rgn);
	   }
	 }

	 for(int j = 0; j < 2; j++) {
	   for(int k = 0; k < 4; k++) {
	     bool fg=(((rctInfo.hfQBits>> (j * 4 + k)) & 0x1)  == 0x1); 
	     L1CaloRegion rgn = L1CaloRegion(rctInfo.hfEt[j][k], fg,  rctInfo.crateID , (j * 4 +  k));
	     rgn.setBx(iBX);
	     colls->rctCalo()->push_back(rgn);
	   }
	 }
       }
     }

}

bool RctRawToDigi::decodeLinkID(const uint32_t inputValue, uint32_t &crateNumber, uint32_t &linkNumber, bool &even)
{
  //if crateNumber not valid set to 0xFF
  crateNumber = ( inputValue >> 8 ) & 0xFF;
  if(crateNumber > 17)
    crateNumber = 0xFF;
  
  //if linkNumber not valid set to 0xFF
  linkNumber = ( inputValue ) & 0xFF;
  
  if(linkNumber > 12)
    linkNumber = 0xFF;
  
  if((linkNumber&0x1) == 0)
    even=true;
  else
    even=false;
  
  return true;
};


bool
RctRawToDigi::printAll(const unsigned char *data, const unsigned size)
{
  for(unsigned i = 0; i < size; i ++){
    std::cout << data[i] << " ";
    if(i%6==5)
      std::cout<<std::endl;
  }
  return true;
}


// ------------ method called once each job just after ending the event loop  ------------
void RctRawToDigi::endJob()
{
  unsigned total=0;
  std::ostringstream os;

  for (unsigned i=0 ; i <= MAX_ERR_CODE ; ++i) {
    total+=errorCounters_.at(i);
    os << "Error " << i << " (" << errorCounters_.at(i) << ")";
    if(i < MAX_ERR_CODE) { os << ", "; }
  }

  if (total>0 && verbose_) {
    edm::LogError("RCT") << "Encountered " << total << " unpacking errors: " << os.str();
  }  
}

/// make this a plugin
DEFINE_FWK_MODULE(RctRawToDigi);
