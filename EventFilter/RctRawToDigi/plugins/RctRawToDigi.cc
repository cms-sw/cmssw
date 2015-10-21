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


RctRawToDigi::RctRawToDigi(const edm::ParameterSet& iConfig) :
  inputLabel_(iConfig.getParameter<edm::InputTag>("inputLabel")),
  fedId_(iConfig.getUntrackedParameter<int>("rctFedId", FEDNumbering::MINTriggerUpgradeFEDID)),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose",false))
{
  LogDebug("RCT") << "RctRawToDigi will unpack FED Id " << fedId_;
  /** Register Products **/
  // RCT collections
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();

  // Error collection
  consumes<FEDRawDataCollection>(inputLabel_);

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

  using namespace edm;

  // Instantiate all the collections the unpacker needs; puts them in event when this object goes out of scope.
  std::auto_ptr<RctUnpackCollections> colls(new RctUnpackCollections(iEvent));  
  
  // get raw data collection
  edm::Handle<FEDRawDataCollection> feds;
  iEvent.getByLabel(inputLabel_, feds);

  // if raw data collection is present, check the headers and do the unpacking
  if (feds.isValid()) {
  
    const FEDRawData& rctRcd = feds->FEDData(fedId_);

    //Check FED size
    LogDebug("RCT") << "Upacking FEDRawData of size " << std::dec << rctRcd.size();

    //check header size
    if ( rctRcd.size() < sLinkHeaderSize_ + sLinkTrailerSize_ + amc13HeaderSize_ + amc13TrailerSize_ + MIN_DATA) {
      LogError("L1T") << "Cannot unpack: empty/invalid L1T raw data (size = "
		      << rctRcd.size() << ") for ID " << fedId_ << ". Returning empty collections!";
      //continue;
      return;
    }

    // do the unpacking
    unpack(rctRcd, iEvent, colls.get()); 

  }
  else{

    LogError("L1T") << "Cannot unpack: no collection found";

    return; 
  }

}

void RctRawToDigi::unpack(const FEDRawData& d, edm::Event& e, RctUnpackCollections * const colls)
{

  const unsigned char * data = d.data();  // The 8-bit wide raw-data array.  
  // Data offset - starts at 16 as there is a 64-bit S-Link header followed
  // by a 64-bit software-controlled header (for pipeline format version
  // info that is not yet used).

  
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
  
  FEDTrailer trailer(data + (d.size() - sLinkTrailerSize_));
  
  if (trailer.check()) {
    LogDebug("L1T") << "Found SLink trailer:"
		    << " Length " << trailer.lenght()
		    << " CRC " << trailer.crc()
		    << " Status " << trailer.evtStatus()
		    << " Throttling bits " << trailer.ttsBits();
  } else {
    LogWarning("L1T") << "Did not find a SLink trailer!";
  }
  
  unpackCTP7((uint32_t*)data, 0, sizeof(data), colls);

}


void
RctRawToDigi::unpackCTP7(const uint32_t *data, const unsigned block_id, const unsigned size, RctUnpackCollections * const colls)
{
  //offset from 6 link header words
  uint32_t of = 6;
  LogDebug("L1T") << "Block ID  = " << block_id << " size = " << size;

  CTP7Format ctp7Format;
  RctDataDecoder rctDataDecoder;
  uint32_t nBXTemp = 0; 
  uint32_t ctp7FWVersion;
  uint32_t L1ID, L1aBCID;
  std::vector<RCTInfo> allCrateRCTInfo[5];
  
  L1ID          =  data[1+of];                      // extract the L1 ID number
  L1aBCID       =  data[5+of] & 0x00000FFF;         // extract the BCID number of L1A
  nBXTemp       = (data[5+of] & 0x00FF0000) >> 16;  // extract number of BXs readout per L1A 
  ctp7FWVersion =  data[4+of];
  
  if(nBXTemp != 1 && nBXTemp != 3 && nBXTemp != 5)
    nBXTemp = 1;

  const uint32_t nBX = nBXTemp;

  LogDebug("L1T") << "CTP7 L1ID = " << L1ID << " L1A BCID = " << L1aBCID << " BXs in capture = " << nBX << " CTP7 DAQ FW = " << ctp7FWVersion;
  
  struct link_data{
    bool even;
    unsigned int crateID;
    unsigned int ctp7LinkNumber;
    std::vector <unsigned int> uint;
  };

  //nBX max 5, nLinks max 36 [nBX][nLinks]
  link_data allLinks[5][36];
  const uint32_t NLinks = ctp7Format.NLINKS;
  assert(NLinks <= 36);

  //change this implementation
  uint32_t iDAQBuffer = 0;

  //Step 1: Grab all data from ctp7 buffer and put into link format
  for(unsigned int iLink = 0; iLink < NLinks; iLink++ ){
    iDAQBuffer = of +
                 ctp7Format.EVENT_HEADER_WORDS + iLink * (ctp7Format.CHANNEL_HEADER_WORDS + 
							  nBX * ctp7Format.CHANNEL_DATA_WORDS_PER_BX);
    
    //first decode linkID 
    uint32_t linkID     = data[iDAQBuffer++];
    uint32_t tmp        = data[iDAQBuffer++];
    uint32_t CRCErrCnt  =  tmp & 0x0000FFFF;
    //uint32_t linkStatus = (tmp & 0xFFFF0000) >> 16;

    uint32_t crateID = 0;    uint32_t expectedCrateID = 0;
    bool even = false; bool expectedEven = false;

    //getExpected Link ID
    rctDataDecoder.getExpectedLinkID(iLink, expectedCrateID, expectedEven);
    //getDecodedLink ID
    rctDataDecoder.decodeLinkID(linkID, crateID, even);

    //Perform a check to see if the link ID is as expected, if not then report an error but continue unpacking
    if(expectedCrateID!=crateID || even!=expectedEven ){
      LogError("L1T") <<"Expected Crate ID "<< expectedCrateID <<" expectedEven "<< expectedEven 
		      <<"does not match actual Crate ID "<<crateID<<" even "<<even;
    }

    if(CRCErrCnt!=0)
      LogError("L1T")<<"WARNING CRC ErrorFound linkID "<< linkID<<" expected crateID "<< expectedCrateID;
    
    // Loop over multiple BX                                                                        
    for (uint32_t iBX=0; iBX<nBX; iBX++){
      allLinks[iBX][iLink].uint.reserve(6);
      allLinks[iBX][iLink].ctp7LinkNumber     = iLink;
      allLinks[iBX][iLink].crateID            = expectedCrateID;
      allLinks[iBX][iLink].even               = expectedEven;

      //Notice 6 words per BX
      for(unsigned int iWord = 0; iWord < 6 ; iWord++ ){
	allLinks[iBX][iLink].uint.push_back(data[iDAQBuffer+iWord+iBX*6]);
      }
    }
  }

  //Step 2: Dynamically match links and create RCTInfo Objects 
  uint32_t nCratesFound = 0;
  for(unsigned int iCrate = 0; iCrate < 18 ; iCrate++){
    
    bool foundEven = false, foundOdd = false;
    link_data even[5];
    link_data odd[5];
    
    for(unsigned int iLink = 0; iLink < NLinks; iLink++){
      
      if( (allLinks[0][iLink].crateID==iCrate) && (allLinks[0][iLink].even == true) ){
	foundEven = true;
	for (unsigned int iBX=0; iBX<nBX; iBX++)
	  even[iBX] = allLinks[iBX][iLink];
      }
      else if( (allLinks[0][iLink].crateID==iCrate) && (allLinks[0][iLink].even == false) ){
	foundOdd = true;
	for (unsigned int iBX=0; iBX<nBX; iBX++)
	  odd[iBX] = allLinks[iBX][iLink];
      } 

      //if success then create RCTInfoVector and fill output object
      if(foundEven && foundOdd){
	nCratesFound++;
	   
	//fill rctInfoVector for all BX read out
	for (unsigned int iBX=0; iBX<nBX; iBX++){
	  //RCTInfoFactory rctInfoFactory;
	  std::vector <RCTInfo> rctInfoData;
	  rctDataDecoder.decodeLinks(even[iBX].uint, odd[iBX].uint, rctInfoData);
	  rctDataDecoder.setRCTInfoCrateID(rctInfoData, iCrate);
	  allCrateRCTInfo[iBX].push_back(rctInfoData.at(0));
	}
	break;
      }
    }
  }

  if(nCratesFound != 18)
    LogError("L1T") << "Warning -- only found "<< nCratesFound << " valid crates";

  //start assuming 1 BX readout
 int32_t startBX = 0;
  if(nBX == 1)
    startBX = 0;
  else if(nBX == 3)
    startBX = -1;
  else if(nBX == 5)
    startBX = -2;  
  
  //Step 3: Create Collections from RCTInfo Objects  
  //Notice, iBX used for grabbing data from array, startBX used for storing in Collection
  for (uint32_t iBX=0; iBX<nBX; iBX++, startBX++){

    for(unsigned int iCrate = 0; iCrate < nCratesFound; iCrate++ ){
      
      RCTInfo rctInfo = allCrateRCTInfo[iBX].at(iCrate);
      //Use Crate ID to identify eta/phi of candidate
      for(int j = 0; j < 4; j++) {
	L1CaloEmCand em = L1CaloEmCand(rctInfo.neRank[j], 
				       rctInfo.neRegn[j], 
				       rctInfo.neCard[j], 
				       rctInfo.crateID, 
				       false);
	em.setBx(startBX);
	colls->rctEm()->push_back(em);
      }
      
      for(int j = 0; j < 4; j++) {
	L1CaloEmCand em = L1CaloEmCand(rctInfo.ieRank[j], 
				       rctInfo.ieRegn[j], 
				       rctInfo.ieCard[j], 
				       rctInfo.crateID, 
				       true);
	em.setBx(startBX);
	colls->rctEm()->push_back(em);
      }
      
      for(int j = 0; j < 7; j++) {
	for(int k = 0; k < 2; k++) {
	  bool o = (((rctInfo.oBits >> (j * 2 + k)) & 0x1) == 0x1);
	  bool t = (((rctInfo.tBits >> (j * 2 + k)) & 0x1) == 0x1);
	  bool m = (((rctInfo.mBits >> (j * 2 + k)) & 0x1) == 0x1);
	  bool q = (((rctInfo.qBits >> (j * 2 + k)) & 0x1) == 0x1);
	  L1CaloRegion rgn = L1CaloRegion(rctInfo.rgnEt[j][k], o, t, m, q, rctInfo.crateID , j, k);
	  rgn.setBx(startBX);
	  colls->rctCalo()->push_back(rgn);
	}
      }
      
      for(int k = 0; k < 4; k++) {
	for(int j = 0; j < 2; j++) {
	  // 0 1 4 5 2 3 6 7
	  uint32_t offset = j*2 + k%2 + (k/2)*4;
	  bool fg=(((rctInfo.hfQBits >> offset) & 0x1)  == 0x1); 
	  L1CaloRegion rgn = L1CaloRegion(rctInfo.hfEt[j][k], fg,  rctInfo.crateID , (j * 4 + k));
	  rgn.setBx(startBX);
	  colls->rctCalo()->push_back(rgn);
	}
      }
    }
  }
}



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
